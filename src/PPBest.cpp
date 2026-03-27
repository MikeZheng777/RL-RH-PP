#include "PPBest.h"
#include <numeric>   // For std::iota
#include <limits>    // For std::numeric_limits
#include <random>    // For random number generation
#include <iostream>
#include <algorithm> 
#include <future>
#include <mutex>

PPBest::PPBest(const BasicGraph &G, SingleAgentSolver &path_planner) : MAPFSolver(G, path_planner){};

PPBest::~PPBest(){};

void PPBest::clear()
{
    starts.clear();
    goal_locations.clear();
    runtime_rt = 0;
    solution_found = false;
    solution_cost = -2;
    avg_path_length = -1;
    paths.clear();
    best_paths.clear();
    rt.clear();

}

string PPBest::vector_to_string(const vector<int>& v) {
    string result;
    for (int num : v) {
        result += std::to_string(num) + ",";
    }
    return result;
}

bool PPBest::find_path() 
{
    double lowest_cost = INFINITY;
    bool best_order_has_fallback = false;
    best_paths.resize(num_of_agents, nullptr);

    if (current_order_group.empty())
    {
        throw std::runtime_error("PPBest: current_order_group is empty. No groups of orders provided.");
    }

    const unsigned int hw = std::thread::hardware_concurrency();
    const size_t max_workers = hw == 0 ? 4 : static_cast<size_t>(hw);
    std::vector<std::future<OrderEvalResult>> futures;
    futures.reserve(max_workers);

    size_t total = current_order_group.size();
    for (size_t batch_start = 0; batch_start < total; batch_start += max_workers)
    {
        size_t batch_end = std::min(batch_start + max_workers, total);
        futures.clear();
        for (size_t idx = batch_start; idx < batch_end; ++idx)
        {
            const auto& order = current_order_group[idx];
            futures.emplace_back(std::async(std::launch::async, [this, order]() {
                return evaluate_order(order);
            }));
        }

        // Collect and choose the best for this batch
        for (size_t idx = batch_start; idx < batch_end; ++idx)
        {
            OrderEvalResult result = futures[idx - batch_start].get();
            runtime = (double)(std::clock() - start) / CLOCKS_PER_SEC;
            if (result.cost < lowest_cost)
            {
                lowest_cost = result.cost;
                best_order = current_order_group[idx];
                best_order_has_fallback = result.has_fallback;

                // Record failed calls for best order
                last_num_failed_calls = result.failed_calls;

                // Deep copy the paths into best_paths
                for (size_t i = 0; i < result.paths.size(); ++i)
                {
                    if (best_paths[i] != nullptr)
                    {
                        delete best_paths[i];
                        best_paths[i] = nullptr;
                    }
                    best_paths[i] = new Path(result.paths[i]);
                }
            }
        }
    }

    // Return false if the best order used any fallback; otherwise return true.
    return !best_order_has_fallback;
}

PPBest::OrderEvalResult PPBest::evaluate_order(const std::vector<int>& total_order)
{
    OrderEvalResult out;
    out.cost = 0.0;
    out.has_fallback = false;
    out.paths.clear();
    out.paths.resize(num_of_agents);
    out.failed_calls = 0;

    ReservationTable local_rt(G);
    local_rt.num_of_agents = num_of_agents;
    local_rt.map_size = G.size();
    local_rt.k_robust = k_robust;
    local_rt.window = window;
    local_rt.hold_endpoints = hold_endpoints;
    local_rt.use_cat = rt.use_cat;
    local_rt.prioritize_start = rt.prioritize_start;

    std::unique_ptr<SingleAgentSolver> local_planner;
    // if (path_planner.getName() == std::string("SIPP"))
    //     local_planner.reset(new SIPP());
    // else
    //     local_planner.reset(new StateTimeAStar());
    local_planner.reset(new SIPP());

    local_planner->travel_times = travel_times;
    local_planner->hold_endpoints = hold_endpoints;
    local_planner->prioritize_start = prioritize_start;
    local_planner->suboptimal_bound = path_planner.suboptimal_bound;

    vector<Path*> local_paths(num_of_agents, nullptr);
    list< pair<int, Path> > local_paths_list;
    int num_failed_calls = 0;

    for (int i : total_order)
    {
        unordered_set<int> higher_priority_agent;
        for (int j : total_order)
        {
            if (j == i) break;
            higher_priority_agent.insert(j);
        }

        Path path;
        double path_cost = 0;
        int start_location = starts[i].location;

        local_rt.build(local_paths, initial_constraints, higher_priority_agent, i, start_location);
        path = local_planner->run(G, starts[i], goal_locations[i], local_rt);
        path_cost = local_planner->path_cost;
        local_rt.clear();

        if (path.empty())
        {
            // Fallback: use the pre-computed shortest path.
            local_paths[i] = &shortest_paths[i];
            num_failed_calls++;
            out.has_fallback = true;
            out.cost += shortest_path_costs[i];
        }
        else
        {
            local_paths_list.emplace_back(i, path);
            local_paths[i] = &local_paths_list.back().second;
            out.cost += path_cost;
        }
    }

    out.cost += 1000 * num_failed_calls;
    out.failed_calls = num_failed_calls;

    // Copy chosen paths into output
    for (int agent = 0; agent < num_of_agents; ++agent)
    {
        if (local_paths[agent] != nullptr)
            out.paths[agent] = *local_paths[agent];
        else
            out.paths[agent] = Path();
    }

    return out;
}



double PPBest::find_path_per_order(const std::vector<int>& total_order, bool fake_order, bool &order_has_fallback)
{
    clock_t time = std::clock();
    double total_path_cost = 0;
    order_has_fallback = false; // flag to indicate fallback occurred
    paths_list.clear();
    int num_faild_call = 0;
    
    for (int i : total_order)
    {   
        unordered_set<int> higher_priority_agent;
        if (!fake_order)
        {
            for (int j : total_order)
            {
                if (j == i)
                    break;
                higher_priority_agent.insert(j);
            }
        }
        
        Path path;
        double path_cost;
        int start_location = starts[i].location;
        clock_t t = std::clock();
        rt.build(paths, initial_constraints, higher_priority_agent, i, start_location);
        runtime_rt += (double)(std::clock() - t) / CLOCKS_PER_SEC;
        
        t = std::clock();
        path = path_planner.run(G, starts[i], goal_locations[i], rt);
        runtime_plan_paths += (double)(std::clock() - t) / CLOCKS_PER_SEC;
        path_cost = path_planner.path_cost;
        rt.clear();

        if (path.empty())
        {   
            // Fallback: use the pre-computed shortest path.
            paths[i] = &shortest_paths[i]; 
            num_faild_call++;
            num_faild_order++;  
            order_has_fallback = true; // mark that this order had a fallback
            // add path cost of the shortest path
            total_path_cost += shortest_path_costs[i] ;
        }
        else
        {
            paths_list.emplace_back(i, path);
            paths[i] = &paths_list.back().second;
            total_path_cost += path_cost;
        }
    }
    
    return total_path_cost + 1000*num_faild_call;
}



bool PPBest::validate_solution()
{
    list<Conflict> conflict;
	for (int a1 = 0; a1 < num_of_agents; a1++)
	{
		for (int a2 = a1 + 1; a2 < num_of_agents; a2++)
		{
            find_conflicts(conflict, a1, a2);
            if (!conflict.empty())
            {
                int a1_, a2_, loc1, loc2, t;
                std::tie(a1_, a2_, loc1, loc2, t) = conflict.front();
                if (loc2 < 0)
                    std::cout << "Agents "  << a1 << " and " << a2 << " collides at " << loc1 <<
                    " at timestep " << t << std::endl;
                else
                    std::cout << "Agents " << a1 << " and " << a2 << " collides at (" <<
                              loc1 << "-->" << loc2 << ") at timestep " << t << std::endl;
                return false;
            }
		}
	}
	return true;
}

void PPBest::find_conflicts(list<Conflict>& conflicts, int a1, int a2)
{
    clock_t t = clock();
    if (paths[a1] == nullptr || paths[a2] == nullptr)
        return;
	
		// TODO: add k-robust

    int size1 = min(window + 1, (int)paths[a1]->size());
    int size2 = min(window + 1, (int)paths[a2]->size());
    for (int timestep = 0; timestep < size1; timestep++)
    {
        if (size2 <= timestep - k_robust)
            break;
        int loc = paths[a1]->at(timestep).location;
        for (int i = max(0, timestep - k_robust); i <= min(timestep + k_robust, size2 - 1); i++)
        {
            if (loc == paths[a2]->at(i).location && G.types[loc] != "Magic")
            {
                conflicts.emplace_back(a1, a2, loc, -1, min(i, timestep)); // k-robust vertex conflict
                // runtime_detect_conflicts += (double)(std::clock() - t) / CLOCKS_PER_SEC;
                return;
            }
        }
        if (k_robust == 0 && timestep < size1 - 1 && timestep < size2 - 1) // detect edge conflicts
        {
            int loc1 = paths[a1]->at(timestep).location;
            int loc2 = paths[a2]->at(timestep).location;
            if (loc1 != loc2 && loc1 == paths[a2]->at(timestep + 1).location
                        && loc2 == paths[a1]->at(timestep + 1).location)
            {
                conflicts.emplace_back(a1, a2, loc1, loc2, timestep + 1); // edge conflict
                // runtime_detect_conflicts += (double)(std::clock() - t) / CLOCKS_PER_SEC;
                return;
            }
        }

    }
    
	// runtime_detect_conflicts += (double)(std::clock() - t) / CLOCKS_PER_SEC;
}

void PPBest::get_solution()
{
    // update_paths(best_node);
    solution.resize(num_of_agents);
    for (int k = 0; k < num_of_agents; k++)
    {
        solution[k] = *best_paths[k];
    }

    //solution_cost  = 0;
    avg_path_length = 0;

    for (int k = 0; k < num_of_agents; k++)
    {
        avg_path_length += best_paths[k]->size();
    }
    avg_path_length /= num_of_agents;
}


bool PPBest::run(const vector<State>& starts,
            const vector< vector<pair<int, int> > >& goal_locations, // an ordered list of pairs of <location, release time>
            double _time_limit)
{
    clear();
    start = std::clock();

    this->starts = starts;
    this->goal_locations = goal_locations;
    this->num_of_agents = starts.size();
    this->time_limit = _time_limit;

    solution_cost = INFINITY;
    solution_found = false;

    rt.num_of_agents = num_of_agents;
    rt.map_size = G.size();
    rt.k_robust = k_robust;
    rt.window = window;
	rt.hold_endpoints = hold_endpoints;
    path_planner.travel_times = travel_times;
	path_planner.hold_endpoints = hold_endpoints;
	path_planner.prioritize_start = prioritize_start;

    // best_paths.resize(num_of_agents,nullptr);

    find_shortest_paths();


    solution_found = find_path();
    get_solution();
    
    if (!solution_found)
    {
        if (screen > 0)
        {
            std::cout << "PPBest failed" << std::endl;
        }   
        return false;
        
    }
    
    min_sum_of_costs = 0;
    for (int i = 0; i < num_of_agents; i++)
    {
        int start_loc = starts[i].location;
        for (const auto& goal : goal_locations[i])
        {
            min_sum_of_costs += G.heuristics.at(goal.first)[start_loc];
            start_loc = goal.first;
        }
    }
	// if (screen > 0) // 1 or 2
	// 	print_results();
	return solution_found;
}