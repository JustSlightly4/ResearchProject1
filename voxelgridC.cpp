// grid.hpp
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <random>
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h> // For bind_vector

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<float>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<float>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<float>>>);

auto wrap_index = [](int idx, int dim) {
    return (idx % dim + dim) % dim; // always in [0, dim-1]
};

class VoxelGridC {
public:
    Eigen::Matrix3d cell; 
    Eigen::Matrix3d cell_inv; 
    Eigen::Vector3i gpts;
    Eigen::Vector3d resolution;
    py::array_t<float> grid;

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
        for (std::size_t i = 0; i < gpts[0]; i++) {
			for (std::size_t j = 0; j < gpts[1]; j++) {
				for (std::size_t k = 0; k < gpts[2]; k++) {
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
        
        Eigen::Vector3d center_frac = (center.transpose() * cell_inv).unaryExpr([](double x) {
			return x - std::floor(x); // wrap into [0,1)
		});
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
        
        Eigen::Vector3d center_frac = (center.transpose() * cell_inv).unaryExpr([](double x) {
			return x - std::floor(x); // wrap into [0,1)
		});
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
        
        Eigen::Vector3d center_frac = (center.transpose() * cell_inv).unaryExpr([](double x) {
			return x - std::floor(x); // wrap into [0,1)
		});
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
        
        Eigen::Vector3d center_frac = (center.transpose() * cell_inv).unaryExpr([](double x) {
			return x - std::floor(x); // wrap into [0,1)
		});
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
    std::vector<Eigen::Vector3d> sample_voxels_in_range(float min_val = 0.0f, float max_val = 1.0f, float min_dist = 0.0f, bool return_indices = false,unsigned int seed = 0) const { 
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

        // Shuffle candidate indices
        std::vector<size_t> indices(candidates.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<Eigen::Vector3d> results;
        float min_dist2 = min_dist * min_dist;

        for (size_t idx : indices) {
            if (return_indices) {
                Eigen::Vector3i ijk = candidates[idx];
                results.emplace_back((float)ijk[0], (float)ijk[1], (float)ijk[2]);
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

        return results;
    }
    
    //NOT ACTUALLY CACHED YET
	py::array_t<bool> cached_sphere_mask(float radius) {
		//The size of the grid
		int nx = gpts(0);
		int ny = gpts(1);
		int nz = gpts(2);
		
		//C++ Only mask declaration
		py::array_t<bool> maskArray({nx, ny, nz});
		auto mask = maskArray.mutable_unchecked<3>();

		// Center of the sphere in fractional coordinates
		Eigen::Vector3d center_frac(0.5, 0.5, 0.5);

		//Getting the mesh grid
		for (int ix = 0; ix < nx; ++ix) {
			for (int iy = 0; iy < ny; ++iy) {
				for (int iz = 0; iz < nz; ++iz) {
					// Fractional coordinates of the current voxel
					Eigen::Vector3d frac_coords(
						(ix + 0.5) / nx,
						(iy + 0.5) / ny,
						(iz + 0.5) / nz
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
		return maskArray;
	}
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
		//.def_property_readonly("cell", [](const VoxelGridC& g) { return g.cell; })
        //.def_property_readonly("cell_inv", [](const VoxelGridC& g) { return g.cell_inv; })
        //.def_property_readonly("gpts", [](const VoxelGridC& g) { return g.gpts; })
        //.def_property_readonly("resolution", [](const VoxelGridC& g) { return g.resolution; })
        .def_readwrite("cell", &VoxelGridC::cell)
        .def_readwrite("cell_inv", &VoxelGridC::cell_inv)
        .def_readwrite("gpts", &VoxelGridC::gpts)
        .def_readwrite("resolution", &VoxelGridC::resolution)
        .def_readwrite("grid", &VoxelGridC::grid);  // read/write access from Python
}
