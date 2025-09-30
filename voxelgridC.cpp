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

namespace py = pybind11;

class VoxelGridC {
public:
    Eigen::Matrix3f cell; //3D matrix
    Eigen::Matrix3f cell_inv; //The 3D matrix inverse 
    Eigen::Vector3i gpts; //3 dimensional vector for the grid size
    Eigen::Vector3f resolution; //3 dimensional vector for the resolution
    std::vector<std::vector<std::vector<float>>> grid;

	//Constructor
    VoxelGridC(const Eigen::Matrix3f& cell_input,
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

        Eigen::Vector3f lengths;
        for (int i = 0; i < 3; ++i)
            lengths[i] = cell.row(i).norm();

        if (resolution > 0.0f) {
            gpts = (lengths / resolution).array().ceil().cast<int>();
            this->resolution = lengths.cwiseQuotient(gpts.cast<float>());
        } else {
            gpts = gpts_input;
            this->resolution = lengths.cwiseQuotient(gpts.cast<float>());
        }

        // Allocate 3D grid
        grid.resize(gpts[0], std::vector<std::vector<float>>(gpts[1], std::vector<float>(gpts[2], 0.0f)));
        std::cout << grid.size() << ", " << grid[0].size() << ", " << grid[0][0].size() << "\n";
    }
    
    //Position_to_index
    Eigen::Vector3i position_to_index(const Eigen::Vector3f& r) const {
        // Map to fractional coordinates
        Eigen::Vector3f frac = cell_inv * r;

        // Wrap into [0,1)
        Eigen::Vector3f wrapped_frac = frac.array() - frac.array().floor();  // frac % 1.0

        // Convert back to wrapped real-space position
        Eigen::Vector3f r_wrapped = cell * wrapped_frac;

        // Convert to final fractional coordinates again
        Eigen::Vector3f frac_wrapped = cell_inv * r_wrapped;

        // Clip to [0, 1) to avoid boundary issues
        for (int i = 0; i < 3; ++i) {
            if (frac_wrapped[i] >= 1.0f)
                frac_wrapped[i] = std::nextafter(1.0f, 0.0f);
            else if (frac_wrapped[i] < 0.0f)
                frac_wrapped[i] = 0.0f;
        }

        // Convert to grid index
        Eigen::Vector3i idx = (frac_wrapped.array() * gpts.cast<float>().array()).floor().cast<int>();
        return idx;
    }

	//Index to Position
    Eigen::Vector3f index_to_position(int i, int j, int k) const {
        Eigen::Vector3f frac = (Eigen::Vector3f(i, j, k) + Eigen::Vector3f::Constant(0.5f)).cwiseQuotient(gpts.cast<float>());
        Eigen::Vector3f r = cell * frac;
        return r;
    }
    
    //std::cout << "Grid[" << i << "][" << j << "][" << k << "] = " << grid[i][j][k] << (grid[i][j][k] == value ? " == " : " != ") << value << "\n";
    //Set Sphere
    void set_sphere(const Eigen::Vector3f& center, float radius, float value = 1.0f) {
        Eigen::Vector3f center_frac = (center.transpose() * cell_inv).unaryExpr([](float x) {
            return x - std::floor(x);
        });

        Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<float>().array()).floor().cast<int>();

		//Set Sphere Mask
		int counter = 0;
        for (int i = 0; i < gpts[0]; ++i) {
            for (int j = 0; j < gpts[1]; ++j) {
                for (int k = 0; k < gpts[2]; ++k) {
                    Eigen::Vector3f frac = (Eigen::Vector3f(i, j, k) + Eigen::Vector3f::Constant(0.5f)).cwiseQuotient(gpts.cast<float>());
                    Eigen::Vector3f disp_frac = frac - center_frac;
                    disp_frac = disp_frac.unaryExpr([](float x) {
                        return x - std::round(x);
                    });

                    Eigen::Vector3f disp_mic = disp_frac.transpose() * cell;
                    float dist2 = disp_mic.squaredNorm();
                    if (dist2 <= radius * radius) {
                        grid[i][j][k] = value;
						++counter;
                    }
                }
            }
        }
        std::cout << "Ran " << counter << " times\n";
    }

	//Add Sphere
    void add_sphere(const Eigen::Vector3f& center, float radius, float value = 1.0f) {
        Eigen::Vector3f center_frac = (center.transpose() * cell_inv).unaryExpr([](float x) {
            return x - std::floor(x);
        });

        Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<float>().array()).floor().cast<int>();

		//Add Sphere Mask
        for (int i = 0; i < gpts[0]; ++i) {
            for (int j = 0; j < gpts[1]; ++j) {
                for (int k = 0; k < gpts[2]; ++k) {
                    Eigen::Vector3f frac = (Eigen::Vector3f(i, j, k) + Eigen::Vector3f::Constant(0.5f)).cwiseQuotient(gpts.cast<float>());
                    Eigen::Vector3f disp_frac = frac - center_frac;
                    disp_frac = disp_frac.unaryExpr([](float x) {
                        return x - std::round(x);
                    });

                    Eigen::Vector3f disp_mic = disp_frac.transpose() * cell;
                    float dist2 = disp_mic.squaredNorm();
                    if (dist2 <= radius * radius) {
                        grid[i][j][k] += value;
                    }
                }
            }
        }
    }
    
    //Multiply Sphere
    void mul_sphere(const Eigen::Vector3f& center, float radius, float factor = 2.0f) {
		// Convert center to fractional coordinates, wrapped into [0,1)
		Eigen::Vector3f center_frac = (center.transpose() * cell_inv).unaryExpr([](float x) {
			return x - std::floor(x);
		});

		// Convert fractional coords to grid index
		Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<float>().array()).floor().cast<int>();

		//Multiply Sphere Mask
		for (int i = 0; i < gpts[0]; ++i) {
			for (int j = 0; j < gpts[1]; ++j) {
				for (int k = 0; k < gpts[2]; ++k) {
					// Compute fractional coordinates of voxel center
					Eigen::Vector3f frac = (Eigen::Vector3f(i, j, k) + Eigen::Vector3f::Constant(0.5f))
											 .cwiseQuotient(gpts.cast<float>());

					// Displacement in fractional coordinates (apply minimum-image convention)
					Eigen::Vector3f disp_frac = frac - center_frac;
					disp_frac = disp_frac.unaryExpr([](float x) {
						return x - std::round(x);
					});

					// Convert to Cartesian displacement
					Eigen::Vector3f disp_mic = disp_frac.transpose() * cell;
					float dist2 = disp_mic.squaredNorm();

					if (dist2 <= radius * radius) {
						grid[i][j][k] *= factor; // multiply instead of add/set
					}
				}
			}
		}
	}
	
	//Divide Sphere
	void div_sphere(const Eigen::Vector3f& center, float radius, float factor = 2.0f) {
		if (factor == 0.0f) {
			throw std::invalid_argument("factor must not be zero");
		}

		// Convert center to fractional coordinates, wrapped into [0,1)
		Eigen::Vector3f center_frac = (center.transpose() * cell_inv).unaryExpr([](float x) {
			return x - std::floor(x);
		});

		// Convert fractional coords to grid index
		Eigen::Vector3i center_idx = (center_frac.array() * gpts.cast<float>().array()).floor().cast<int>();

		//Divide Sphere mask
		for (int i = 0; i < gpts[0]; ++i) {
			for (int j = 0; j < gpts[1]; ++j) {
				for (int k = 0; k < gpts[2]; ++k) {
					// Compute fractional coordinates of voxel center
					Eigen::Vector3f frac = (Eigen::Vector3f(i, j, k) + Eigen::Vector3f::Constant(0.5f))
											 .cwiseQuotient(gpts.cast<float>());

					// Displacement in fractional coordinates (minimum-image convention)
					Eigen::Vector3f disp_frac = frac - center_frac;
					disp_frac = disp_frac.unaryExpr([](float x) {
						return x - std::round(x);
					});

					// Convert to Cartesian displacement
					Eigen::Vector3f disp_mic = disp_frac.transpose() * cell;
					float dist2 = disp_mic.squaredNorm();

					if (dist2 <= radius * radius) {
						grid[i][j][k] /= factor; // divide instead of multiply
					}
				}
			}
		}
	}
	
	// Clamp all voxel values to [min_val, max_val]
	void clamp_grid(float min_val = 0.0f, float max_val = 1.0f) {
		for (int i = 0; i < gpts[0]; ++i) {
			for (int j = 0; j < gpts[1]; ++j) {
				for (int k = 0; k < gpts[2]; ++k) {
					float& v = grid[i][j][k];
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
    std::vector<Eigen::Vector3f> sample_voxels_in_range(float min_val = 0.0f, float max_val = 1.0f, float min_dist = 0.0f, bool return_indices = false,unsigned int seed = 0) const { 
        // Collect candidates
        std::vector<Eigen::Vector3i> candidates;
        for (int i = 0; i < gpts[0]; ++i) {
            for (int j = 0; j < gpts[1]; ++j) {
                for (int k = 0; k < gpts[2]; ++k) {
                    float val = grid[i][j][k];
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
        std::vector<Eigen::Vector3f> positions;
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

        std::vector<Eigen::Vector3f> results;
        float min_dist2 = min_dist * min_dist;

        for (size_t idx : indices) {
            if (return_indices) {
                Eigen::Vector3i ijk = candidates[idx];
                results.emplace_back((float)ijk[0], (float)ijk[1], (float)ijk[2]);
            } else {
                const Eigen::Vector3f& pos = positions[idx];

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
};


PYBIND11_MODULE(voxelgridC, m) {
    py::class_<VoxelGridC>(m, "VoxelGridC")
        .def(py::init<const Eigen::Matrix3f&, float, const Eigen::Vector3i&>(),
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
        .def_property_readonly("grid", [](const VoxelGridC& g) {
			auto& grid = g.grid;
			py::ssize_t nx = grid.size();
			py::ssize_t ny = grid[0].size();
			py::ssize_t nz = grid[0][0].size();

			// Create a NumPy array with shape (nx, ny, nz)
			py::array_t<double> arr({nx, ny, nz});
			auto buf = arr.mutable_unchecked<3>();  // 3D array access

			for (py::ssize_t i = 0; i < nx; i++)
				for (py::ssize_t j = 0; j < ny; j++)
					for (py::ssize_t k = 0; k < nz; k++)
						buf(i, j, k) = grid[i][j][k];

			return arr;
		});
}
