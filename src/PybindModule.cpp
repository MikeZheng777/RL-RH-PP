#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "RLEnvironment.h"


namespace py = pybind11;

PYBIND11_MODULE(warehouse_sim, m) {
    py::class_<RLEnvironment>(m, "RLEnvironment")
        .def(py::init<const py::kwargs&>())
        .def("reset", &RLEnvironment::reset)
        .def("step", &RLEnvironment::step)
        .def("get_observation", &RLEnvironment::getObservation)
        .def("get_reward", &RLEnvironment::getReward)
        .def("is_done", &RLEnvironment::isDone)
        .def("get_info", &RLEnvironment::getInfo)
        .def("get_solve_time", &RLEnvironment::get_solve_time)
        .def_readwrite("config", &RLEnvironment::kwargs_); 
       
         // Expose non-const attribute
        // get the arritbuts of warehouse_system.solve time of RLEnvironment



}
