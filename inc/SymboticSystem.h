#pragma once
#include "BasicSystem.h"
#include "SymboticGraph.h"

class SymboticSystem:
	public BasicSystem
{
public:
	SymboticSystem(const SymboticGrid& G, MAPFSolver& solver);
	~SymboticSystem();

	void simulate(int simulation_time);
    vector<vector<int> > loading_track;
    void printGoalLocations() const;
	void step();

private:
	const SymboticGrid& G;
	unordered_set<int> held_endpoints;

	void initialize();
	void initialize_start_locations();
	void initialize_goal_locations();
	void update_goal_locations();
    int assign_inbound_station() const;
    int assign_outbound_station() const;
    int assign_aisle_station() const;

};

