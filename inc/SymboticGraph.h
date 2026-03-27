#pragma once
#include "BasicGraph.h"


class SymboticGrid:
	public BasicGraph
{
public:
	// vector<int> endpoints;
    vector<int> inbounds;
    vector<int> outbounds;
    vector<int> asiles;
	vector<int> agent_home_locations;
    
    bool load_map(string fname);
    void preprocessing(bool consider_rotation); // compute heuristics
private:
    // bool load_weighted_map(string fname);
    bool load_unweighted_map(string fname);
};
