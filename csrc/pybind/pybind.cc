// SYSTEM INCLUDES
// #include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <pybind11/functional.h>


// C++ PROJECT INCLUDES
#include "dbscan/dbscan.h"


std::function<void(void)> empty_func = []() -> void {};

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.def("euclidean_dist", &euclidean_dist);
    m.def("expand_cluster", &expand_cluster);
    m.def("dbscan_train", &dbscan_train, py::arg("X"), py::arg("epsilon"), py::arg("min_pts"),
                                         py::arg("assignment_vec"), py::arg("monitor_func"));

};

