# PFM_EX_1

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/kalpeshsingh89/PFM_EX_1)](https://github.com/kalpeshsingh89/PFM_EX_1)
[![Last Commit](https://img.shields.io/github/last-commit/kalpeshsingh89/PFM_EX_1)](https://github.com/kalpeshsingh89/PFM_EX_1/commits/main)
[![Stars](https://img.shields.io/github/stars/kalpeshsingh89/PFM_EX_1?style=social)](https://github.com/kalpeshsingh89/PFM_EX_1/stargazers)
[![Issues](https://img.shields.io/github/issues/kalpeshsingh89/PFM_EX_1)](https://github.com/kalpeshsingh89/PFM_EX_1/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/kalpeshsingh89/PFM_EX_1)](https://github.com/kalpeshsingh89/PFM_EX_1/pulls)

Phase-Field Fracture Mechanics Example using [deal.II](https://www.dealii.org/).  
This repository contains a modified version of **step-19.cc** adapted for phase-field fracture simulations with adaptive mesh refinement.

---

## 🔧 Requirements
- **deal.II** ≥ 9.4  
- **CMake** ≥ 3.10  
- **C++17 compiler** (GCC, Clang, or Intel)  
- Optional: **MPI** and **PETSc/Trilinos** for large-scale parallel runs

---

## ⚙️ Build Instructions
```bash
# Clone the repo
git clone https://github.com/kalpeshsingh89/PFM_EX_1.git
cd PFM_EX_1/trial_1

# Create build folder
mkdir build && cd build

# Configure with deal.II
cmake -DDEAL_II_DIR=/path/to/dealii ..

# Compile
make -j4



▶️ Run Simulation
From inside build/:
./step-19
Simulation results (solution_u-*.vtu, solution_phiH-*.vtu, results.pvd) will be written in the project folder and can be opened in ParaView

📊 Outputs
solution_u-*.vtu → Displacement field
solution_phiH-*.vtu → Phase-field damage variable
results.pvd → ParaView collection file

🚀 Next Steps
Extend to 2D/3D fracture simulations
Explore different phase-field length scales
Enable adaptive mesh refinement (AMR)
Benchmark vs. XFEM results

📄 License
MIT License (or specify your choice).

✍️ Author
Developed by Kalpesh Singh as part of computational mechanics research in fracture modeling with deal.II.
