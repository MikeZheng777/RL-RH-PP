#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boost/filesystem.hpp>
#include <ctime>
#include <stdexcept>
#include "RLEnvironment.h"
#include "KivaSystem.h"
#include "SymboticSystem.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <csignal>
#include <execinfo.h>

namespace py = pybind11;

// Global base output directory for logging from signal handlers
static std::string g_base_outfile;

static void append_cpp_exception_log(const std::string& base_outfile, const std::string& msg)
{
    namespace fs = boost::filesystem;

    // Determine target log path with robust fallbacks
    std::string log_path;
    bool fallback_to_cwd = base_outfile.empty() || base_outfile == "None";
    if (!fallback_to_cwd)
    {
        try {
            fs::path out_dir(base_outfile);
            if (!fs::exists(out_dir)) {
                // attempt to create the output directory tree
                fs::create_directories(out_dir);
            }
            if (fs::exists(out_dir) && fs::is_directory(out_dir)) {
                log_path = (out_dir / "cpp_exceptions.log").string();
            } else {
                fallback_to_cwd = true;
            }
        } catch (...) {
            fallback_to_cwd = true;
        }
    }
    if (fallback_to_cwd) {
        log_path = std::string("cpp_exceptions.log");
    }

    std::ofstream ofs(log_path, std::ios::app);
    if (!ofs.is_open()) return;
    std::time_t t = std::time(nullptr);
    ofs << "[" << std::asctime(std::localtime(&t)) << "] " << msg << std::endl;
}

// Minimal, async-signal-safe-ish handler to record fatal signals to log
static void rl_signal_handler(int sig)
{
    const char* name = (sig == SIGSEGV ? "SIGSEGV" : (sig == SIGABRT ? "SIGABRT" : (sig == SIGTERM ? "SIGTERM" : "SIGNAL")));
    try {
        std::string header = std::string("[FATAL] Caught ") + name + " (" + std::to_string(sig) + ")";
        append_cpp_exception_log(g_base_outfile, header);
        // Capture a simple backtrace (best-effort)
        void* buffer[64];
        int nptrs = backtrace(buffer, 64);
        if (nptrs > 0) {
            char** symbols = backtrace_symbols(buffer, nptrs);
            if (symbols != nullptr) {
                std::ostringstream oss;
                oss << "[BACKTRACE] frames=" << nptrs;
                append_cpp_exception_log(g_base_outfile, oss.str());
                for (int i = 0; i < nptrs; ++i) {
                    append_cpp_exception_log(g_base_outfile, std::string("  ") + symbols[i]);
                }
            }
        }
    } catch (...) {}
    std::_Exit(128 + sig);
}

RLEnvironment::RLEnvironment(const py::kwargs& kwargs)
    : kwargs_(kwargs), warehouse_system(nullptr)  // Initialize system to nullptr
{
    py::gil_scoped_acquire acquire; 
    // initialize();
    // set_parameters();
    // warehouse_system->simulation_time = simulation_time;
    // warehouse_system->initialize();
}

RLEnvironment::~RLEnvironment() 
{
    py::gil_scoped_acquire acquire;

    // Manually clean up the dynamically allocated system
    if (warehouse_system != nullptr) 
    {
        delete warehouse_system;
        warehouse_system = nullptr;
    }
    if (solver_) {
        delete solver_;
        solver_ = nullptr;
    }
    if (kiva_grid_) {
        delete kiva_grid_;
        kiva_grid_ = nullptr;
    }
    if (symbotic_grid_) {
        delete symbotic_grid_;
        symbotic_grid_ = nullptr;
    }
}

void RLEnvironment::reset(int seed)
{
    py::gil_scoped_acquire acquire;  // Acquire GIL

    try
    {
        // Reset the environment to the initial state
        if (warehouse_system != nullptr)
        {
            delete warehouse_system;  // Clean up existing system before reinitializing
            warehouse_system = nullptr;
        }
        if (solver_) {
        delete solver_;
        solver_ = nullptr;
        }
        if (kiva_grid_) {
            delete kiva_grid_;
            kiva_grid_ = nullptr;
        }
        if (symbotic_grid_) {
            delete symbotic_grid_;
            symbotic_grid_ = nullptr;
        }

        initialize();
        set_parameters(seed);
        warehouse_system->initialize();

        if (kwargs_["screen"].cast<int>() > 0)
        {
            std::cout << "*** Simulating " << warehouse_system->seed << " ***" << std::endl;
        }
    } 
    catch (const std::exception &e) 
    {
        std::string base = warehouse_system ? warehouse_system->outfile : std::string();
        append_cpp_exception_log(base, std::string("[ERROR] Reset failed: ") + e.what());
        throw py::value_error(std::string("[ERROR] Reset failed: ") + e.what());
    } 
    catch (...) 
    {
        std::string base = warehouse_system ? warehouse_system->outfile : std::string();
        append_cpp_exception_log(base, std::string("[ERROR] Reset failed: Unknown exception occurred."));
        throw py::value_error("[ERROR] Reset failed: Unknown exception occurred.");
    }
}

void RLEnvironment::initialize() 
{
    // py::gil_scoped_release release;
    py::gil_scoped_acquire acquire;
    std::string scenario = kwargs_["scenario"].cast<std::string>();
    std::string map_file = kwargs_["map"].cast<std::string>();

    if (scenario == std::string("KIVA")) 
    {
        kiva_grid_ = new KivaGrid();
        if (!kiva_grid_->load_map(map_file)) 
        {
            delete kiva_grid_;  // Clean up if load_map fails
            throw std::runtime_error("Failed to load map for KIVA scenario");
        }
        solver_ = set_solver(*kiva_grid_);
        warehouse_system = new KivaSystem(*kiva_grid_, *solver_);
        warehouse_system->consider_rotation = kwargs_["rotation"].cast<bool>();

        kiva_grid_->preprocessing(warehouse_system->consider_rotation);
        map_size = kiva_grid_->size();

    } else if (scenario == std::string("SYMBOTIC")) 
    {
        symbotic_grid_ = new SymboticGrid();
        if (!symbotic_grid_->load_map(map_file)) {
            delete symbotic_grid_;
            throw std::runtime_error("Failed to load map for SYMBOTIC scenario");
        }
        solver_ = set_solver(*symbotic_grid_);
        warehouse_system = new SymboticSystem(*symbotic_grid_, *solver_);
        warehouse_system->consider_rotation = kwargs_["rotation"].cast<bool>();

        symbotic_grid_->preprocessing(warehouse_system->consider_rotation);
        map_size = symbotic_grid_->size();
        // std::cout << "map_size: " << map_size << std::endl;
    } else 
    {
        throw std::runtime_error("Unknown scenario: " + scenario);
    }
    
}

void RLEnvironment::set_parameters(int seed)
{
    py::gil_scoped_acquire acquire;
    if (!warehouse_system)
    {
        throw std::runtime_error("System not initialized");
    }

    warehouse_system->outfile = kwargs_["output"].cast<std::string>();
    g_base_outfile = warehouse_system->outfile;
    warehouse_system->screen = kwargs_["screen"].cast<int>();
    warehouse_system->log = kwargs_["log"].cast<bool>();
    warehouse_system->num_of_drives = kwargs_["agent_num"].cast<int>();
    warehouse_system->simulation_time = kwargs_["simulation_time"].cast<int>();
    warehouse_system->time_limit = kwargs_["cutoff_time"].cast<double>();
    warehouse_system->simulation_window = kwargs_["simulation_window"].cast<int>();
    warehouse_system->planning_window = kwargs_["planning_window"].cast<int>();
    warehouse_system->low_level_planning_window = kwargs_["low_level_planning_window"].cast<int>();
    warehouse_system->k_robust = kwargs_["robust"].cast<int>();
    warehouse_system->travel_time_window = kwargs_["travel_time_window"].cast<int>();
    warehouse_system->hold_endpoints = kwargs_["hold_endpoints"].cast<bool>();
    warehouse_system->useDummyPaths = kwargs_["dummy_paths"].cast<bool>();

    observation_window = kwargs_["observation_window"].cast<int>();

    // if (kwargs_.contains("seed"))
    // {
    //     warehouse_system->seed = kwargs_["seed"].cast<int>();
    // }
    // else
    // {
    //     warehouse_system->seed = static_cast<int>(std::time(nullptr));
    // }

    warehouse_system->seed = seed;

    std::srand(warehouse_system->seed);

    create_directories();
    check_parameters();

    // Install signal handlers to capture any fatal terminations into the log
    std::signal(SIGSEGV, rl_signal_handler);
    std::signal(SIGABRT, rl_signal_handler);
    std::signal(SIGTERM, rl_signal_handler);
}

// void RLEnvironment::set_seed(int seed) {
//         warehouse_system->seed = seed;
//     }

MAPFSolver* RLEnvironment::set_solver(const BasicGraph& G)
{
    py::gil_scoped_acquire acquire;

    std::string signle_agent_solver_name = kwargs_["single_agent_solver"].cast<std::string>();
    SingleAgentSolver* path_planner = nullptr;
    MAPFSolver* mapf_solver = nullptr;

    if (signle_agent_solver_name == std::string("ASTAR")) 
    {
        path_planner = new StateTimeAStar();
    } 
    else if (signle_agent_solver_name == std::string("SIPP")) 
    {
        path_planner = new SIPP();
    } 
    else 
    {
        throw std::runtime_error("Single-agent solver " + signle_agent_solver_name + " does not exist!");
    }

    std::string solver_name = kwargs_["solver"].cast<std::string>();
    // if (solver_name == std::string("ECBS")) 
    // {
    //     ECBS* ecbs = new ECBS(G, *path_planner);
    //     ecbs->potential_function = kwargs_["potential_function"].cast<std::string>();
    //     ecbs->potential_threshold = kwargs_["potential_threshold"].cast<double>();
    //     ecbs->suboptimal_bound = kwargs_["suboptimal_bound"].cast<double>();
    //     mapf_solver = ecbs;
    // } 
    // else if (solver_name == std::string("PBS")) 
    // {
    //     PBS* pbs = new PBS(G, *path_planner);
    //     pbs->lazyPriority = kwargs_["lazyP"].cast<bool>();
    //     bool prioritize_start = kwargs_["prioritize_start"].cast<bool>();
    //     if (kwargs_["hold_endpoints"].cast<bool>() || kwargs_["dummy_paths"].cast<bool>())
    //         prioritize_start = false;
    //     pbs->prioritize_start = prioritize_start;
    //     pbs->setRT(kwargs_["CAT"].cast<bool>(), prioritize_start);
    //     mapf_solver = pbs;
    // }
    // else if (solver_name == std::string("PP"))
    if (solver_name == std::string("PPStep"))
    {
        PPStep* pp = new PPStep(G, *path_planner);
		// pp->num_order_sample =1;
        bool prioritize_start = kwargs_["prioritize_start"].cast<bool>();
        // if (vm["hold_endpoints"].as<bool>() or vm["dummy_paths"].as<bool>())
        //     prioritize_start = false;
		prioritize_start = false;
        pp->prioritize_start = prioritize_start;
        pp->setRT(kwargs_["CAT"].cast<bool>(), prioritize_start);
		mapf_solver = pp;
    }
    else if (solver_name == std::string("PPBest"))
    {
        PPBest* pp = new PPBest(G, *path_planner);
        bool prioritize_start = kwargs_["prioritize_start"].cast<bool>();
        // if (vm["hold_endpoints"].as<bool>() or vm["dummy_paths"].as<bool>())
        //     prioritize_start = false;
		prioritize_start = false;
        pp->prioritize_start = prioritize_start;
        pp->setRT(kwargs_["CAT"].cast<bool>(), prioritize_start);
		mapf_solver = pp;
    }
    else if (solver_name == std::string("PBS"))
    {
        PBS* pbs = new PBS(G, *path_planner);
        bool prioritize_start = kwargs_["prioritize_start"].cast<bool>();
        // if (vm["hold_endpoints"].as<bool>() or vm["dummy_paths"].as<bool>())
        //     prioritize_start = false;
		prioritize_start = true;
        pbs->prioritize_start = prioritize_start;
        pbs->setRT(kwargs_["CAT"].cast<bool>(), prioritize_start);
		mapf_solver = pbs;
    }
    else if (solver_name == std::string("ECBS"))
    {
        ECBS* ecbs = new ECBS(G, *path_planner);
		ecbs->potential_function = kwargs_["potential_function"].cast<string>();
		ecbs->potential_threshold = kwargs_["potential_threshold"].cast<double>();
		ecbs->suboptimal_bound = kwargs_["suboptimal_bound"].cast<double>();
		mapf_solver = ecbs;
    }
    else
    {
        throw std::runtime_error("RLEnvironment: Invalid selection of MAPF solver: " + solver_name);
    }
    // Additional solver handling here...

    return mapf_solver;
}

vector<int> RLEnvironment::convert_action_to_vector(const py::array_t<int>& action) 
{
    // py::gil_scoped_acquire acquire;
    py::buffer_info buf = action.request();
    if (buf.ndim != 1) 
    {
    throw std::runtime_error("Expected prioirity order 1D action array");
    }

    int* ptr = static_cast<int*>(buf.ptr);
    vector<int> action_vector(ptr, ptr + buf.size);
    return action_vector;
}

vector<vector<int>> RLEnvironment::convert_action_to_group_vector(const py::array_t<int>& action) 
{
    py::buffer_info buf = action.request();
    if (buf.ndim != 2) 
    {
        throw std::runtime_error("Expected a 2D numpy array for group of priority orders");
    }

    size_t num_orders = buf.shape[0];
    size_t order_length = buf.shape[1];

    // Pointer to the raw data.
    int* ptr = static_cast<int*>(buf.ptr);
    vector<vector<int>> group_vector;
    group_vector.reserve(num_orders);

    for (size_t i = 0; i < num_orders; i++) 
    {
        vector<int> order;
        order.reserve(order_length);
        for (size_t j = 0; j < order_length; j++) 
        {
            order.push_back(ptr[i * order_length + j]);
        }
        group_vector.push_back(std::move(order));
    }
    
    return group_vector;
}


py::tuple RLEnvironment::step(const py::array_t<int>& action) 
{
    py::gil_scoped_acquire acquire;
    if (!warehouse_system) 
    {
        throw std::runtime_error("System not initialized");
    }
    std::string solver_name = kwargs_["solver"].cast<std::string>();
    try
    {
        if (solver_name == std::string("PPBest"))
        {
            vector<vector<int>> action_vector_group = convert_action_to_group_vector(action);
            warehouse_system->solver.current_order_group = action_vector_group;
        }
        else if (solver_name == std::string("PPStep"))
        {
            vector<int> action_vector = convert_action_to_vector(action);
            warehouse_system->solver.current_order = action_vector;
        }
        
        // warehouse_system->solver.find_shortest_paths();
        warehouse_system->step();
        warehouse_system->timestep += warehouse_system->simulation_window;  // Simulate for one timestep
        return py::make_tuple(getObservation(), getReward(), isDone(), getInfo());
    }
    catch (const std::exception& e)
    {
        append_cpp_exception_log(warehouse_system ? warehouse_system->outfile : std::string(), std::string("[ERROR] Step failed: ") + e.what());
        throw py::value_error(std::string("[ERROR] Step failed: ") + e.what());
    }
    catch (...)
    {
        append_cpp_exception_log(warehouse_system ? warehouse_system->outfile : std::string(), std::string("[ERROR] Step failed: Unknown exception."));
        throw py::value_error("[ERROR] Step failed: Unknown exception.");
    }
}

py::tuple RLEnvironment::getObservation() 
{

    py::gil_scoped_acquire acquire;  // Acquire the GIL before interacting with Python
    if (!warehouse_system) 
    {
        throw std::runtime_error("System not initialized");
    }

    if (warehouse_system->solver.shortest_paths.empty()) 
    {
        throw std::runtime_error("No paths available in the solver.");
    }

    // Create a py::list to store all paths
    py::list paths_list;

    // Iterate over all paths in warehouse_system->solver.shortest_paths
    for (const auto& path : warehouse_system->solver.shortest_paths) 
    {
        // Create a vector to store the flattened data of this path
        std::vector<int> path_data;

        // Iterate over each state in the current path
        for (const State& state : path) 
        {
            // Add state properties to path_data (assuming State has location, timestep, and orientation)
            path_data.push_back(static_cast<int>(state.location));
            path_data.push_back(static_cast<int>(state.timestep));
            path_data.push_back(static_cast<int>(state.orientation));
        }

        // Convert the path_data to a NumPy array and add it to the paths_list
        paths_list.append(py::array_t<int>(
            py::array::ShapeContainer({static_cast<long int>(path.size()), 3}),  // Shape: (path length, 3)
            path_data.data()                                                     // Data pointer
        ));
    }

    // Return the paths_list as a tuple or directly as a list
    return py::make_tuple(paths_list);
}


py::tuple RLEnvironment::getReward() 
{
    py::gil_scoped_acquire acquire;
    if (!warehouse_system) 
    {
        throw std::runtime_error("System not initialized");
    }
    int num_new_finished_task = warehouse_system->num_new_finished_task;
    bool solver_success_flag = warehouse_system->solver_success_flag;
    // bool congested_flag = warehouse_system->congested();
    bool congested_flag = false;
    // double reward = warehouse_system->solver.avg_path_length;
    double reward_1 = warehouse_system->avg_solution_length;
    double reward_2 = warehouse_system->avg_num_collision_solver;
    double reward_3 = warehouse_system->avg_dis_to_goal_after_excuted;
    double reward_4 = warehouse_system->ratio_wait_agent_solver;
    

    return py::make_tuple(num_new_finished_task, solver_success_flag, congested_flag, reward_1, reward_2, reward_3, reward_4);
    // Implement actual reward logic here

}

bool RLEnvironment::isDone() 
{
    py::gil_scoped_acquire acquire;
    if (!warehouse_system) 
    {
        throw std::runtime_error("System not initialized");
    }
    // Implement actual termination condition here

    // if (warehouse_system.congested() || warehouse_system->timestep >= warehouse_system.simulation_time)
    if (warehouse_system->timestep >= warehouse_system->simulation_time)
    {
        if (kwargs_["save_visialization"].cast<bool>())
        {
            warehouse_system->save_results();
        }
        return true;
    }
    return false;
}

py::dict RLEnvironment::getInfo() 
{
    py::gil_scoped_acquire acquire;
    if (!warehouse_system) 
    {
        throw std::runtime_error("System not initialized");
    }

    py::dict info;
    // Implement actual info logic here
    if (kwargs_["solver"].cast<std::string>() == std::string("PPBest"))
    {
        std::vector<int> best_order = warehouse_system->solver.best_order;
        py::list best_order_list = py::cast(best_order);
        info["best_order"] = best_order_list;
        // expose number of failed low-level calls of the best order
        try {
            // dynamic cast to access PPBest-specific field
            auto* ppbest = dynamic_cast<PPBest*>(&warehouse_system->solver);
            if (ppbest != nullptr)
            {
                info["num_failed_calls"] = ppbest->last_num_failed_calls;
            }
        } catch (...) {}
    }

    py::list paths_list;

    // Iterate over all paths in warehouse_system->solver.shortest_paths
    for (const auto& path : warehouse_system->solver.solution) 
    {
        // Create a vector to store the flattened data of this path
        std::vector<int> path_data;

        // Iterate over each state in the current path in next 5 steps
        // for (const State& state : path) 
        for (size_t i = 0; i < path.size() && i < warehouse_system->simulation_window; ++i) 
        {
            const State& state = path[i];

            // Add state properties to path_data (assuming State has location, timestep, and orientation)
            path_data.push_back(static_cast<int>(state.location));
            path_data.push_back(static_cast<int>(state.timestep));
            path_data.push_back(static_cast<int>(state.orientation));
        }

        // Convert the path_data to a NumPy array and add it to the paths_list
        paths_list.append(py::array_t<int>(
            py::array::ShapeContainer({static_cast<long int>(path.size()), 3}),  // Shape: (path length, 3)
            path_data.data()                                                     // Data pointer
        ));
    }
    // Convert the paths_list to a NumPy array and add it to the info dictionary
    info["executed_paths"] = paths_list;

    return info;
}

void RLEnvironment::create_directories() 
{
    py::gil_scoped_acquire acquire;
    boost::filesystem::path dir(kwargs_["output"].cast<std::string>() + "/");
    boost::filesystem::create_directories(dir);
    if (kwargs_["log"].cast<bool>()) 
    {
        boost::filesystem::path dir1(kwargs_["output"].cast<std::string>() + "/goal_nodes/");
        boost::filesystem::path dir2(kwargs_["output"].cast<std::string>() + "/search_trees/");
        boost::filesystem::create_directories(dir1);
        boost::filesystem::create_directories(dir2);
    }
}

void RLEnvironment::check_parameters() 
{
    py::gil_scoped_acquire acquire;
    if (kwargs_["hold_endpoints"].cast<bool>() && kwargs_["dummy_paths"].cast<bool>()) 
    {
        throw std::runtime_error("Hold endpoints and dummy paths cannot be used simultaneously");
    }
    if ((kwargs_["hold_endpoints"].cast<bool>() || kwargs_["dummy_paths"].cast<bool>()) &&
        kwargs_["simulation_window"].cast<int>() != 1) 
    {
        throw std::runtime_error("Hold endpoints and dummy paths can only work when the simulation window is 1");
    }
    if ((kwargs_["hold_endpoints"].cast<bool>() || kwargs_["dummy_paths"].cast<bool>()) &&
        kwargs_["planning_window"].cast<int>() < INT_MAX / 2) 
    {
        throw std::runtime_error("Hold endpoints and dummy paths cannot work with planning windows");
    }
}