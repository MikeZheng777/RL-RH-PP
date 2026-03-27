#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "BasicSystem.h"
#include "MAPFSolver.h"
#include "BasicGraph.h"
#include "KivaGraph.h"
#include "SymboticGraph.h"

namespace py = pybind11;

class RLEnvironment {
public:
    RLEnvironment(const py::kwargs& kwargs);
    ~RLEnvironment();  // Destructor to clean up raw pointers
    void reset(int seed);
    py::tuple step(const py::array_t<int>& action);
    py::tuple getObservation();
    py::tuple getReward();
    bool isDone();
    py::dict getInfo();
    int observation_window = 5;
    py::kwargs kwargs_;

    double get_solve_time() { return warehouse_system->solve_time; }
    

private:
    void initialize();
    void set_parameters(int seed);
    MAPFSolver* set_solver(const BasicGraph& G);  // Return raw pointer now
    void create_directories();
    void check_parameters();
    vector<int> convert_action_to_vector(const py::array_t<int>& action);
    vector<vector<int>> convert_action_to_group_vector(const py::array_t<int>& action) ;
    int map_size = 0;

    BasicSystem* warehouse_system;  // Raw pointer for system
    KivaGrid* kiva_grid_ = nullptr;
    SymboticGrid* symbotic_grid_ = nullptr;
    MAPFSolver* solver_ = nullptr;
    
};
