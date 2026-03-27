#include "PPStep.h"
#include "PathTable.h"
#include <numeric>   // For std::iota
#include <limits>    // For std::numeric_limits
#include <random>    // For random number generation
#include <iostream>




PPStep::PPStep(const BasicGraph &G, SingleAgentSolver &path_planner) : MAPFSolver(G, path_planner){};

PPStep::~PPStep(){};

void PPStep::clear()
{
    starts.clear();
    goal_locations.clear();
    runtime_rt = 0;
    solution_found = false;
    solution_cost = -2;
    avg_path_length = -1;
    paths.clear();
    // best_paths.clear();
    rt.clear();

}

string PPStep::vector_to_string(const vector<int>& v) {
    string result;
    for (int num : v) {
        result += std::to_string(num) + ",";
    }
    return result;
}

bool PPStep::find_path()
{


    // int lowest_cost = std::numeric_limits<int>::max();
    // vector<int> current_order(num_of_agents);
    // std::iota(current_order.begin(), current_order.end(), 0);
    // std::unordered_set<std::string> unique_orders; // To track generated orders

    // Random number generator for shuffling
    // std::random_device rd;
    // std::mt19937 g(rd());  

    // while (std::isinf(solution_cost) && runtime < time_limit)
    // {
    //     rt.clear();
    //     // paths.clear();
    //     paths.resize(num_of_agents, nullptr);

    //     do {
    //         std::shuffle(current_order.begin(), current_order.end(), g);
    //     } while (unique_orders.find(vector_to_string(current_order)) != unique_orders.end());

    //     // Store the new unique order
    //     unique_orders.insert(vector_to_string(current_order));
    //     solution_cost = find_path_per_order(current_order, false);
    //     runtime = (double)(std::clock() - start) / CLOCKS_PER_SEC;
    // }  
    paths.resize(num_of_agents, nullptr);
    vector<int> current_order(num_of_agents);
    std::iota(current_order.begin(), current_order.end(), 0);
    // sort the current_order based on the distance of current location to goal
    // close agents get high priority (ascending distance)
    std::sort(current_order.begin(), current_order.end(), [&](int a, int b) {
        return G.heuristics.at(goal_locations[a].back().first)[starts[a].location] <
               G.heuristics.at(goal_locations[b].back().first)[starts[b].location];
    });
    solution_cost = find_path_per_order(current_order, false);
    runtime = (double)(std::clock() - start) / CLOCKS_PER_SEC;
    if (!std::isinf(solution_cost))
    {
        return true;
    }
    else
    {
        return false;
    }
}



double PPStep::find_path_per_order(const std::vector<int> &total_order, bool fake_order)
{
    clock_t time = std::clock();
    double total_path_cost = 0 ;
    paths_list.clear();
    bool use_shortest_path = false;
    for (int i:total_order)
    {   
        unordered_set<int> higher_prioirty_agent;
        if (!fake_order)
        {
            for (int j:total_order)
            {
                if (j==i)
                {
                    break;
                }
                higher_prioirty_agent.insert(j);
            }
        }
        
        Path path;
        double path_cost;
        int start_location = starts[i].location;
        clock_t t = std::clock();
        rt.build(paths, initial_constraints, higher_prioirty_agent, i, start_location);


        runtime_rt += (double)(std::clock() - t) / CLOCKS_PER_SEC;
        t = std::clock();
        path = path_planner.run(G, starts[i], goal_locations[i], rt);
        runtime_plan_paths += (double)(std::clock()-t) / CLOCKS_PER_SEC;
        path_cost = path_planner.path_cost;
        rt.clear();

        

        if (path.empty())
        {   
            paths[i] = &shortest_paths[i]; // use the shortest path for agent i if failed to find a path (will be repaired)
            use_shortest_path = true;
            total_path_cost += shortest_path_costs[i];
        }
        // dummy_start -> paths.emplace_back(i, path);
        else
        {
            paths_list.emplace_back(i, path);
            paths[i] = &paths_list.back().second;
            total_path_cost += path_cost;
        }
        
        
    }
   
    if (use_shortest_path)
    {
        return INFINITY;
    }
    return total_path_cost;
    
}


bool PPStep::validate_solution()
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
                if (screen>0)
                {
                    if (loc2 < 0)
                    {
                        std::cout << "Agents "  << a1 << " and " << a2 << " collides at " << loc1 <<
                        " at timestep " << t << std::endl;

                    }
                    else
                    {
                        std::cout << "Agents " << a1 << " and " << a2 << " collides at (" <<
                                loc1 << "-->" << loc2 << ") at timestep " << t << std::endl;
                    }
                }
                
                return false;
            }
		}
	}
	return true;
}

void PPStep::find_conflicts(list<Conflict>& conflicts, int a1, int a2)
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

void PPStep::get_solution()
{
    // update_paths(best_node);
    solution.resize(num_of_agents);
    for (int k = 0; k < num_of_agents; k++)
    {
        solution[k] = *paths[k];
    }

    //solution_cost  = 0;
    avg_path_length = 0;

    for (int k = 0; k < num_of_agents; k++)
    {
        avg_path_length += paths[k]->size();
    }
    avg_path_length /= num_of_agents;

    // avg_path_length = - avg_path_length;
}


bool PPStep::run(const vector<State>& starts,
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
        // if (screen>0)
        // {
        //     std::cout << "PPStep failed" << std::endl;
        // }
        if (screen > 0)
        {   
            failed_times++;
            std::cout << "PPStep failed" << std::endl;
            std::cout << "Failed times: " << failed_times << std::endl;
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