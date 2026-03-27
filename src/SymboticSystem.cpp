#include "SymboticSystem.h"
#include <stdlib.h>
#include "PBS.h"
#include <boost/tokenizer.hpp>
#include "WHCAStar.h"
#include "ECBS.h"
#include "LRAStar.h"


SymboticSystem::SymboticSystem(const SymboticGrid& G, MAPFSolver& solver): BasicSystem(G, solver), G(G) {}


SymboticSystem::~SymboticSystem() {}


void SymboticSystem::initialize_start_locations()
{
    // int N = G.size();
    int N_home = G.agent_home_locations.size();
    std::vector<bool> used(N_home, false);

    // Choose random start locations
    // Any non-obstacle locations can be start locations
    // Start locations should be unique
	for (int k = 0; k < num_of_drives;)
	{
		int idx = rand() % N_home;
        int loc = G.agent_home_locations[idx];
		// if (G.types[loc] = "Home" && !used[loc])
        if (!used[idx])
		{
			int orientation = -1;
			if (consider_rotation)
			{
				orientation = rand() % 4;
			}
			starts[k] = State(loc, 0, orientation);
			paths[k].emplace_back(starts[k]);
			used[idx] = true;
			finished_tasks[k].emplace_back(loc, 0);
			k++;
		}
	}
}


void SymboticSystem::initialize_goal_locations()
{
	if (hold_endpoints || useDummyPaths)
		return;
    // Choose random goal locations
    // a close induct location can be a goal location, or
    // any eject locations can be goal locations
    // Goal locations are not necessarily unique
    for (int k = 0; k < num_of_drives; k++)
    {
		int goal;
		// if (k % 4 == 0) // to inbound
		// {
		// 	goal = assign_inbound_station(starts[k].location);
		// 	// drives_in_induct_stations[goal]++;
		// }
        // else if (k % 5 == 0) // to outbound
        // {
        //     goal = assign_outbound_station(starts[k].location);
        // }
		// else // to asile
		// {
		// 	goal = assign_aisle_station();
		// }
        goal = assign_aisle_station();
		goal_locations[k].emplace_back(goal, 0);
        if (rand() % 2 == 0)
        {
            loading_track[k].emplace_back(0); // loading after the goal is arrived, 0 is without loading, 1 is with loading
        }
        else
        {
            loading_track[k].emplace_back(1);
        }
    }
}


void SymboticSystem::update_goal_locations()
{
	for (int k = 0; k < num_of_drives; k++)
	{
		pair<int, int> curr(paths[k][timestep].location, timestep); // current location

		if (goal_locations[k].empty())
		{
			int next = assign_aisle_station();
			goal_locations[k].emplace_back(next, 0);
			loading_track[k].emplace_back(0);
		}

		pair<int, int> goal = goal_locations[k].back();
		int loading = loading_track[k].back();
		int min_timesteps = G.get_Manhattan_distance(curr.first, goal.first); // cannot use h values, because graph edges may have weights  // G.heuristics.at(goal)[curr];
		// min_timesteps = max(min_timesteps, goal.second);
		while (min_timesteps <= simulation_window)
			// The agent might finish its tasks during the next planning horizon
		{
			// assign a new task
			int next;
			if (G.types[goal.first] == "Inbound")
			{
				next = assign_aisle_station();
                loading_track[k].emplace_back(0);
			}
			else if (G.types[goal.first] == "Outbound")
			{
                if (rand()%2 == 0)
                {
                    next = assign_inbound_station();
                }
                else
                {
                    next = assign_aisle_station();
                }
				
                loading_track[k].emplace_back(1);
				// drives_in_induct_stations[next]++; // the drive will go to the next induct station
			}
            else if (G.types[goal.first]=="Asile")
            {
                if (loading == 0)
                {
                    if (rand()%2 == 0)
                    {
                        next = assign_inbound_station();
                    }
                    else
                    {
                        next = assign_aisle_station();
                    }
                    
                    loading_track[k].emplace_back(1);
                }
                else
                {
                    next = assign_outbound_station();
                    loading_track[k].emplace_back(0);
                }
                
            }
			else
			{
				next = assign_aisle_station();
				loading_track[k].emplace_back(0);
			}
			goal_locations[k].emplace_back(next, 0);
			min_timesteps += G.get_Manhattan_distance(next, goal.first); // G.heuristics.at(next)[goal];
			// min_timesteps = max(min_timesteps, goal.second);
			goal = make_pair(next, 0);
			loading = loading_track[k].back();
		}
	}
}


int SymboticSystem::assign_aisle_station() const
{
    int assigned_aisle;
    int N_asiles = G.asiles.size();
    int idx = rand() % N_asiles;
    assigned_aisle = G.asiles[idx];

    return assigned_aisle;
}

int SymboticSystem::assign_inbound_station() const
{
    int assigned_inbound;
    int N_inbound = G.inbounds.size();
    int idx = rand() % N_inbound;
    assigned_inbound = G.inbounds[idx];

    return assigned_inbound;
}

int SymboticSystem::assign_outbound_station() const
{
    int assigned_outbound;
    int N_outbound = G.outbounds.size();
    int idx = rand() % N_outbound;
    assigned_outbound = G.outbounds[idx];

    return assigned_outbound;
}



// int SymboticSystem::assign_induct_station(int curr) const
// {
//     int assigned_loc;
// 	double min_cost = DBL_MAX;
// 	for (auto induct : drives_in_induct_stations)
// 	{
// 		double cost = G.heuristics.at(induct.first)[curr] + c * induct.second;
// 		if (cost < min_cost)
// 		{
// 			min_cost = cost;
// 			assigned_loc = induct.first;
// 		}
// 	}
//     return assigned_loc;
// }


// int SymboticSystem::assign_eject_station() const
// {
// 	int n = rand() % G.ejects.size();
// 	boost::unordered_map<std::string, std::list<int> >::const_iterator it = G.ejects.begin();
// 	std::advance(it, n);
// 	int p = rand() % it->second.size();
// 	auto it2 = it->second.begin();
// 	std::advance(it2, p);
// 	return *it2;
// }

void SymboticSystem::step()
{
	{
    if (screen>0)
    {
        std::cout << "Timestep " << timestep << std::endl;

    }

    update_start_locations();
    update_goal_locations();
    double start_time = std::clock();
    solve();
	double end_time = std::clock();
	// solve_time = (end_time - start_time) / CLOCKS_PER_SEC;

    auto new_finished_tasks = move();
    if (screen>0)
    {
        std::cout << new_finished_tasks.size() << " tasks have been finished" << std::endl;
    }

    num_new_finished_task = new_finished_tasks.size();

    for (auto task : new_finished_tasks)
    {
        int id, loc, t;
        std::tie(id, loc, t) = task;
        finished_tasks[id].emplace_back(loc, t);
        num_of_tasks++;
    }
}
}

void SymboticSystem::simulate(int simulation_time)
{
    std::cout << "*** Simulating " << seed << " ***" << std::endl;
    this->simulation_time = simulation_time;
    initialize();
	
	for (; timestep < simulation_time; timestep += simulation_window)
	{
		// std::cout << "Timestep " << timestep << std::endl;

		// update_start_locations();
		// update_goal_locations();
        // // this->printGoalLocations();
        // // std::cout << goal_locations << "goal_locations"<< std::endl;
		// solve();

		// // move drives
		// auto new_finished_tasks = move();
		// std::cout << new_finished_tasks.size() << " tasks has been finished" << std::endl;

		// // update tasks
		// for (auto task : new_finished_tasks)
		// {
		// 	int id, loc, t;
		// 	std::tie(id, loc, t) = task;
		// 	finished_tasks[id].emplace_back(loc, t);
		// 	num_of_tasks++;
		// 	// if (G.types[loc] == "Induct")
		// 	// {
		// 	// 	drives_in_induct_stations[loc]--; // the drive will leave the current induct station
		// 	// }
		// }
		step();
		
		

		// if (congested())
		// {
		// 	cout << "***** Too many traffic jams ***" << endl;
		// 	break;
		// }
	}

    update_start_locations();
    std::cout << std::endl << "Done!" << std::endl;
    save_results();
}


void SymboticSystem::initialize()
{
	initialize_solvers();

	starts.resize(num_of_drives);
	goal_locations.resize(num_of_drives);
	paths.resize(num_of_drives);
	finished_tasks.resize(num_of_drives);
    loading_track.resize(num_of_drives);

	// for (const auto induct : G.inducts)
	// {
	// 	drives_in_induct_stations[induct.second] = 0;
	// }

	// bool succ = load_records(); // continue simulating from the records
	bool succ = false;
	if (!succ)
	{
		timestep = 0;
		succ = load_locations();
		if (!succ)
		{
			// cout << "Randomly generating initial locations" << endl;
			initialize_start_locations();
			initialize_goal_locations();
		}
	}

	// // initialize induct station counter
	// for (int k = 0; k < num_of_drives; k++)
	// {
	// 	// goals
	// 	int goal = goal_locations[k].back().first;
	// 	if (G.types[goal] == "Induct")
	// 	{
	// 		drives_in_induct_stations[goal]++;
	// 	}
	// 	else if (G.types[goal] != "Eject")
	// 	{
	// 		std::cout << "ERROR in the type of goal locations" << std::endl;
	// 		std::cout << "The fiducial type of the goal of agent " << k << " is " << G.types[goal] << std::endl;
	// 		exit(-1);
	// 	}
	// }
}

void SymboticSystem::printGoalLocations() const {
    std::cout << "goal_locations: [" << std::endl;
    for (const auto& vec : goal_locations) {
        std::cout << "  [";
        for (const auto& p : vec) {
            std::cout << "(" << p.first << ", " << p.second << ")";
            if (&p != &vec.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (&vec != &goal_locations.back()) {
            std::cout << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}
