#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <random>
#include <algorithm>
#include <chrono>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

//C++ and Python's % operator work differently
//so this is a work around
auto wrap_index = [](int idx, int dim) {
    return (idx % dim + dim) % dim; // always in [0, dim-1]
};

//The Actual VoxelGridC Class
class EXPORT VoxelGridC {
public:
    Eigen::Matrix3d cell; 
    Eigen::Matrix3d cell_inv; 
    Eigen::Vector3i gpts;
    Eigen::Vector3d resolution;
    py::array_t<float> grid;
    static constexpr size_t MAX_CACHE_SIZE = 50; //Max cache size
    mutable std::list<int> lru_order;  //Tracks usage order
    mutable std::unordered_map<int,
		std::pair<py::array_t<bool>, 
		std::list<int>::iterator>> sphere_mask_cache;
	static constexpr float RADIUS_EPS = 1e-5;

	//Constructor
    VoxelGridC(const Eigen::Matrix3d& cell_input,
              float resolution = -1.0f,
              const Eigen::Vector3i& gpts_input = Eigen::Vector3i(0, 0, 0))
    {
        this->cell = cell_input;
        this->cell_inv = cell.inverse();

		//If the resolution and the gpts is not provided throw error
        if (resolution <= 0.0f && gpts_input == Eigen::Vector3i(0, 0, 0)) {
            throw std::invalid_argument("Either resolution or gpts must be specified");
        }
        
        //If the resolution and the gpts are both provided throw error
        if (resolution > 0.0f && gpts_input != Eigen::Vector3i(0, 0, 0)) {
            throw std::invalid_argument("Only one of resolution or gpts can be specified");
        }

        Eigen::Vector3d lengths;
        for (int i = 0; i < 3; ++i)
            lengths[i] = cell.row(i).norm();

        if (resolution > 0.0f) {
            gpts = (lengths / resolution).array().ceil().cast<int>();
            this->resolution = lengths.cwiseQuotient(gpts.cast<double>());
        } else {
            gpts = gpts_input;
            this->resolution = lengths.cwiseQuotient(gpts.cast<double>());
        }

        // Allocate 3D grid
        grid.resize({gpts[0], gpts[1], gpts[2]});
        
        //Can not modify the grid directly you have to use what can be
        //thought about as a pointer every time.
        auto g = grid.mutable_unchecked<3>();
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < gpts[0]; i++) {
			for (int j = 0; j < gpts[1]; j++) {
				for (int k = 0; k < gpts[2]; k++) {
					g(i, j, k) = 0.0f;
				}
			}
		}
	}
    
    //Position_to_index
    Eigen::Vector3i position_to_index(const Eigen::Vector3d& r) const {
        // Map to fractional coordinates
        Eigen::Vector3d frac = cell_inv * r;

        // Wrap into [0,1)
        Eigen::Vector3d wrapped_frac = frac.array() - frac.array().floor();  // frac % 1.0

        // Convert back to wrapped real-space position
        Eigen::Vector3d r_wrapped = cell * wrapped_frac;

        // Convert to final fractional coordinates again
        Eigen::Vector3d frac_wrapped = cell_inv * r_wrapped;

        // Clip to [0, 1) to avoid boundary issues
        for (int i = 0; i < 3; ++i) {
            if (frac_wrapped[i] >= 1.0f)
                frac_wrapped[i] = std::nextafter(1.0f, 0.0f);
            else if (frac_wrapped[i] < 0.0f)
                frac_wrapped[i] = 0.0f;
        }

        // Convert to grid index
        Eigen::Vector3i idx = (frac_wrapped.array() * gpts.cast<double>().array()).floor().cast<int>();
        return idx;
    }

	//Index to Position
    Eigen::Vector3d index_to_position(int i, int j, int k) const {
        Eigen::Vector3d frac = (Eigen::Vector3d(i, j, k) + Eigen::Vector3d::Constant(0.5f)).cwiseQuotient(gpts.cast<double>());
        Eigen::Vector3d r = cell * frac;
        return r;
    }
    
    //Set Sphere
    void set_sphere(const Eigen::Vector3d& center, float radius, float value = 1.0f) {
		auto grid = this->grid.mutable_unchecked<3>();
        
        Eigen::Vector3d center_frac = (cell_inv * center).array().floor().matrix();
		center_frac = (cell_inv * center) - center_frac; // wrap into [0,1)
		Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<double>().array()).floor().cast<int>();
		py::array_t<bool> maskArray = cached_sphere_mask(radius);
        auto mask = maskArray.mutable_unchecked<3>();
        
        int nx = grid.shape(0);
        int ny = grid.shape(1);
        int nz = grid.shape(2);
        int mx = mask.shape(0);
        int my = mask.shape(1);
        int mz = mask.shape(2);
        int ox = int(mx * 0.5);
        int oy = int(my * 0.5);
        int oz = int(mz * 0.5);
        
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < mx; ++i) {
			for (int j = 0; j < my; ++j) {
				for (int k = 0; k < mz; ++k) {
					if (mask(i, j, k)) {
						int x = wrap_index(center_idx[0] + i - ox, nx);
						int y = wrap_index(center_idx[1] + j - oy, ny);
						int z = wrap_index(center_idx[2] + k - oz, nz);
						grid(x, y, z) = value;
					}
				}
			}
		}
    }
    
    //Add Sphere
    void add_sphere(const Eigen::Vector3d& center, float radius, float value = 1.0f) {
        auto grid = this->grid.mutable_unchecked<3>();
        
        Eigen::Vector3d center_frac = (cell_inv * center).array().floor().matrix();
		center_frac = (cell_inv * center) - center_frac; // wrap into [0,1)
		Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<double>().array()).floor().cast<int>();
		py::array_t<bool> maskArray = cached_sphere_mask(radius);
        auto mask = maskArray.mutable_unchecked<3>();
        
        int nx = grid.shape(0);
        int ny = grid.shape(1);
        int nz = grid.shape(2);
        int mx = mask.shape(0);
        int my = mask.shape(1);
        int mz = mask.shape(2);
        int ox = int(mx * 0.5);
        int oy = int(my * 0.5);
        int oz = int(mz * 0.5);
        
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < mx; ++i) {
			for (int j = 0; j < my; ++j) {
				for (int k = 0; k < mz; ++k) {
					if (mask(i, j, k)) {
						int x = wrap_index(center_idx[0] + i - ox, nx);
						int y = wrap_index(center_idx[1] + j - oy, ny);
						int z = wrap_index(center_idx[2] + k - oz, nz);
						grid(x, y, z) += value;
					}
				}
			}
		}
    }
    
    //Multiply Sphere
    void mul_sphere(const Eigen::Vector3d& center, float radius, float factor = 2.0f) {
		auto grid = this->grid.mutable_unchecked<3>();
        
        Eigen::Vector3d center_frac = (cell_inv * center).array().floor().matrix();
		center_frac = (cell_inv * center) - center_frac; // wrap into [0,1)
		Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<double>().array()).floor().cast<int>();
		py::array_t<bool> maskArray = cached_sphere_mask(radius);
        auto mask = maskArray.mutable_unchecked<3>();
        
        int nx = grid.shape(0);
        int ny = grid.shape(1);
        int nz = grid.shape(2);
        int mx = mask.shape(0);
        int my = mask.shape(1);
        int mz = mask.shape(2);
        int ox = int(mx * 0.5);
        int oy = int(my * 0.5);
        int oz = int(mz * 0.5);
        
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < mx; ++i) {
			for (int j = 0; j < my; ++j) {
				for (int k = 0; k < mz; ++k) {
					if (mask(i, j, k)) {
						int x = wrap_index(center_idx[0] + i - ox, nx);
						int y = wrap_index(center_idx[1] + j - oy, ny);
						int z = wrap_index(center_idx[2] + k - oz, nz);
						grid(x, y, z) *= factor;
					}
				}
			}
		}
	}
	
	//Divide Sphere
	void div_sphere(const Eigen::Vector3d& center, float radius, float factor = 2.0f) {
		auto grid = this->grid.mutable_unchecked<3>();
        
        Eigen::Vector3d center_frac = (cell_inv * center).array().floor().matrix();
		center_frac = (cell_inv * center) - center_frac; // wrap into [0,1)
		Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<double>().array()).floor().cast<int>();
		py::array_t<bool> maskArray = cached_sphere_mask(radius);
        auto mask = maskArray.mutable_unchecked<3>();
        
        int nx = grid.shape(0);
        int ny = grid.shape(1);
        int nz = grid.shape(2);
        int mx = mask.shape(0);
        int my = mask.shape(1);
        int mz = mask.shape(2);
        int ox = int(mx * 0.5);
        int oy = int(my * 0.5);
        int oz = int(mz * 0.5);
        
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < mx; ++i) {
			for (int j = 0; j < my; ++j) {
				for (int k = 0; k < mz; ++k) {
					if (mask(i, j, k)) {
						int x = wrap_index(center_idx[0] + i - ox, nx);
						int y = wrap_index(center_idx[1] + j - oy, ny);
						int z = wrap_index(center_idx[2] + k - oz, nz);
						grid(x, y, z) /= factor;
					}
				}
			}
		}
	}
	
	// Clamp all voxel values to [min_val, max_val]
	void clamp_grid(float min_val = 0.0f, float max_val = 1.0f) {
		auto grid = this->grid.mutable_unchecked<3>();
		#pragma omp parallel for collapse(3) schedule(static)
		for (int i = 0; i < gpts[0]; ++i) {
			for (int j = 0; j < gpts[1]; ++j) {
				for (int k = 0; k < gpts[2]; ++k) {
					float& v = grid(i, j, k);
					if (v < min_val) {
						v = min_val;
					} else if (v > max_val) {
						v = max_val;
					}
				}
			}
		}
	}
    
    //Sample Voxels in Range
    std::vector<Eigen::Vector3d> sample_voxels_in_range(float min_val = 0.0f, float max_val = 1.0f, float min_dist = 0.0f, bool return_indices = false, unsigned int seed = 0) const { 
        auto grid = this->grid.unchecked<3>();
        // Collect candidates
        std::vector<Eigen::Vector3i> candidates;
        for (int i = 0; i < gpts[0]; ++i) {
            for (int j = 0; j < gpts[1]; ++j) {
                for (int k = 0; k < gpts[2]; ++k) {
                    float val = grid(i, j, k);
                    if (val >= min_val && val <= max_val) {
                        candidates.emplace_back(i, j, k);
                    }
                }
            }
        }

        if (candidates.empty()) {
            throw std::runtime_error("No voxels in specified value range.");
        }
        if (return_indices && min_dist > 0.0f) {
            throw std::invalid_argument("min_dist only supported when return_indices=false");
        }

        // Precompute positions if needed
        std::vector<Eigen::Vector3d> positions;
        if (!return_indices) {
            positions.reserve(candidates.size());
            for (auto& idx : candidates) {
                positions.push_back(index_to_position(idx[0], idx[1], idx[2]));
            }
        }

		//Set the seed
        std::mt19937 rng;
        if (seed == 0) {
			seed = static_cast<unsigned>(
				std::chrono::steady_clock::now().time_since_epoch().count()
			);
		}
		rng.seed(seed);
        
        // Shuffle candidate indices
        std::vector<size_t> indices(candidates.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<Eigen::Vector3d> results;
        float min_dist2 = min_dist * min_dist;

        for (size_t idx : indices) {
            if (return_indices) {
                Eigen::Vector3i ijk = candidates[idx];
                results.emplace_back((double)ijk[0], (double)ijk[1], (double)ijk[2]);
            } else {
                const Eigen::Vector3d& pos = positions[idx];

                if (min_dist > 0.0f) {
                    bool too_close = false;
                    for (const auto& sel : results) {
                        if ((sel - pos).squaredNorm() < min_dist2) {
                            too_close = true;
                            break;
                        }
                    }
                    if (too_close) continue;
                }
                results.push_back(pos);
            }
        }
        
        /*
        // Now convert to py::array_t<float>
		int N = temp_results.size();
		py::array_t<float> results({N, 3});
		auto r = results.mutable_unchecked<2>();

		for (ssize_t i = 0; i < N; ++i) {
			r(i, 0) = (float)temp_results[i][0];
			r(i, 1) = (float)temp_results[i][1];
			r(i, 2) = (float)temp_results[i][2];
		}
		*/

        return results;
    }
    
	py::array_t<bool> cached_sphere_mask(float radius) {
		
		//Check Cache
		int key = static_cast<int>(radius / RADIUS_EPS); //float are prone to error so make 
		auto it = sphere_mask_cache.find(key);
        if (it != sphere_mask_cache.end()) {
			//std::cout << "Cache Hit!" << "\n";
            lru_order.erase(it->second.second);
            lru_order.push_front(key);
            it->second.second = lru_order.begin();
            return it->second.first;
        }
        //std::cout << "Cache Miss!" << "\n";
        
		//int nx = gpts[0];
		//int ny = gpts[1];
		//int nz = gpts[2];
		//C++ Only mask declaration
		py::array_t<bool> maskArray({gpts[0], gpts[1], gpts[2]});
		auto mask = maskArray.mutable_unchecked<3>();

		// Center of the sphere in fractional coordinates
		Eigen::Vector3d center_frac(0.5, 0.5, 0.5);

		//Getting the mesh grid
		#pragma omp parallel for collapse(3) schedule(static)
		for (int ix = 0; ix < gpts[0]; ++ix) {
			for (int iy = 0; iy < gpts[1]; ++iy) {
				for (int iz = 0; iz < gpts[2]; ++iz) {
					// Fractional coordinates of the current voxel
					Eigen::Vector3d frac_coords(
						(ix + 0.5) / gpts[0],
						(iy + 0.5) / gpts[1],
						(iz + 0.5) / gpts[2]
					);

					// Displacement vector in fractional coordinates
					Eigen::Vector3d disp_frac = frac_coords - center_frac;

					// Apply minimum image convention (wrap into [-0.5, 0.5))
					disp_frac -= disp_frac.array().round().matrix();

					// Convert to Cartesian coordinates
					Eigen::Vector3d disp_cart = cell * disp_frac;

					// Squared distance
					double dist2 = disp_cart.squaredNorm();

					mask(ix, iy, iz) = (dist2 <= radius * radius);
				}
			}
		}
		// --- Insert into cache ---
        lru_order.push_front(key);
        sphere_mask_cache[key] = {maskArray, lru_order.begin()};

        // --- Evict oldest if over capacity ---
        if (sphere_mask_cache.size() > MAX_CACHE_SIZE) {
            int old_key = lru_order.back();
            lru_order.pop_back();
            sphere_mask_cache.erase(old_key);
        }
		return maskArray;
	}
	/*
	py::array_t<bool> cached_sphere_mask(float radius) {
		
		//Check Cache
		auto it = sphere_mask_cache.find(radius);
        if (it != sphere_mask_cache.end()) {
            lru_order.erase(it->second.second);
            lru_order.push_front(radius);
            it->second.second = lru_order.begin();
            return it->second.first;
        }
        
        int diam_x = static_cast<int>(std::ceil(2 * radius / resolution[0])) + 1;
		int diam_y = static_cast<int>(std::ceil(2 * radius / resolution[1])) + 1;
		int diam_z = static_cast<int>(std::ceil(2 * radius / resolution[2])) + 1;
		//C++ Only mask declaration
		py::array_t<bool> maskArray({diam_x, diam_y, diam_z});
		auto mask = maskArray.mutable_unchecked<3>();

		// Center of the sphere in fractional coordinates
		Eigen::Vector3d center_frac(0.5, 0.5, 0.5);

		//Getting the mesh grid
		#pragma omp parallel for collapse(3) schedule(static)
		for (int ix = 0; ix < diam_x; ++ix) {
			for (int iy = 0; iy < diam_y; ++iy) {
				for (int iz = 0; iz < diam_z; ++iz) {
					// Fractional coordinates of the current voxel
					Eigen::Vector3d frac_coords(
						(ix + 0.5) / diam_x,
						(iy + 0.5) / diam_y,
						(iz + 0.5) / diam_z
					);

					// Displacement vector in fractional coordinates
					Eigen::Vector3d disp_frac = frac_coords - center_frac;

					// Apply minimum image convention (wrap into [-0.5, 0.5))
					disp_frac -= disp_frac.array().round().matrix();

					// Convert to Cartesian coordinates
					Eigen::Vector3d disp_cart = cell * disp_frac;

					// Squared distance
					double dist2 = disp_cart.squaredNorm();

					mask(ix, iy, iz) = (dist2 <= radius * radius);
				}
			}
		}
		// --- Insert into cache ---
        lru_order.push_front(radius);
        sphere_mask_cache[radius] = {maskArray, lru_order.begin()};

        // --- Evict oldest if over capacity ---
        if (sphere_mask_cache.size() > MAX_CACHE_SIZE) {
            float old_radius = lru_order.back();
            lru_order.pop_back();
            sphere_mask_cache.erase(old_radius);
        }
		return maskArray;
	}
	*/
};


PYBIND11_MODULE(voxelgridC, m) {
	namespace py = pybind11;
	//py::bind_vector<std::vector<float>>(m, "VectorFloat");
    //py::bind_vector<std::vector<std::vector<float>>>(m, "VectorVectorFloat");
    //py::bind_vector<std::vector<std::vector<std::vector<float>>>>(m, "Vector3DFloat");
    py::class_<VoxelGridC>(m, "VoxelGridC")
        .def(py::init<const Eigen::Matrix3d&, float, const Eigen::Vector3i&>(),
			py::arg("cell"),
			py::arg("resolution") = -1.0f,
			py::arg("gpts") = Eigen::Vector3i(0, 0, 0))
        .def("add_sphere", &VoxelGridC::add_sphere,
			py::arg("center"),
			py::arg("radius"),
			py::arg("value"))
		.def("mul_sphere", &VoxelGridC::mul_sphere,
			 py::arg("center"),
			 py::arg("radius"),
			 py::arg("factor") = 2.0f)
		.def("div_sphere", &VoxelGridC::div_sphere,
			 py::arg("center"),
			 py::arg("radius"),
			 py::arg("factor") = 2.0f)
		.def("clamp_grid", &VoxelGridC::clamp_grid,
			 py::arg("min_val") = 0.0f,
			 py::arg("max_val") = 1.0f)
        .def("set_sphere", &VoxelGridC::set_sphere,
			py::arg("center"),
			py::arg("radius"),
			py::arg("value"))
        .def("index_to_position", &VoxelGridC::index_to_position,
			py::arg("i"),
			py::arg("j"),
			py::arg("k"))
        .def("position_to_index", &VoxelGridC::position_to_index,
			py::arg("r"))
        .def("sample_voxels_in_range", &VoxelGridC::sample_voxels_in_range,
			py::arg("min_val") = 0.0f,
			py::arg("max_val") = 1.0f,
			py::arg("min_dist") = 0.0f,
			py::arg("return_indices") = false,
			py::arg("seed") = 0)
		.def_property_readonly("cell", [](const VoxelGridC& g) { return g.cell; })
        .def_property_readonly("cell_inv", [](const VoxelGridC& g) { return g.cell_inv; })
        .def_property_readonly("gpts", [](const VoxelGridC& g) { return g.gpts; })
        .def_property_readonly("resolution", [](const VoxelGridC& g) { return g.resolution; })
        //.def_readwrite("cell", &VoxelGridC::cell)
        //.def_readwrite("cell_inv", &VoxelGridC::cell_inv)
        //.def_readwrite("gpts", &VoxelGridC::gpts)
        //.def_readwrite("resolution", &VoxelGridC::resolution)
        .def_readwrite("grid", &VoxelGridC::grid);  // read/write access from Python
}
