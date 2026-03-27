#pragma once
#include "MAPFSolver.h"
// #include "PBSNode.h"

class PPStep:public MAPFSolver 
{
public:
    PPStep(const BasicGraph &G, SingleAgentSolver &path_planner);
    ~PPStep();
    double runtime_rt = 0;
    double runtime_plan_paths = 0;
    
    // vector<int> current_order;
    // int num_order_sample = 0;
    string get_name() const {return "PP"; };

    bool run(const vector<State>& starts,
            const vector< vector<pair<int, int> > >& goal_locations, // an ordered list of pairs of <location, release time>
            double _time_limit);

    // void save_results(const std::string &fileName, const std::string &instanceName) const;
    void save_results(const std::string &fileName, const std::string &instanceName) const {};

	void save_search_tree(const std::string &fileName) const {}
	void save_constraints_in_goal_node(const std::string &fileName) const {}


	// void save_results(const std::string &fileName, const std::string &instanceName) const;
    void clear();
    bool prioritize_start = false;

    void setRT(bool use_cat, bool prioritize_start)
	{
		rt.use_cat = use_cat;
		rt.prioritize_start = prioritize_start;
	}

    int failed_times = 0;


private:
    std::clock_t start = 0;
    vector< Path* > paths;
    // vector<Path*> best_paths;
    list< pair<int, Path> > paths_list;
    

    // PBSNode* dummy_start = nullptr;
    // vector<int> best_order;
    // std::vector<vector<int>> total_orders;
    bool find_path();
    // vector<int> select_best_order();

    double find_path_per_order(const std::vector<int> &total_order, bool fake_order);

    void find_conflicts(list<Conflict>& conflicts, int a1, int a2);
    bool validate_solution();
    string vector_to_string(const vector<int>& v);
    void get_solution();

    bool all_elements_nullptr(const std::vector<Path*>& vec) {
    return std::all_of(vec.begin(), vec.end(), [](Path* ptr) {
        return ptr == nullptr;
    });
    }

    

};