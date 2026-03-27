#pragma once
#include "MAPFSolver.h"
// #include "PBSNode.h"
// #include "common.h"

class PPBest:public MAPFSolver 
{
public:
    PPBest(const BasicGraph &G, SingleAgentSolver &path_planner);
    ~PPBest();
    double runtime_rt = 0;
    double runtime_plan_paths = 0;
    int num_order_sample = 300;
    // vector<int> best_order;
    string get_name() const {return "PPBest"; };

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

    int num_faild_order = 0;
    int last_num_failed_calls = 0; // best order's fallback count in last run


private:
    struct OrderEvalResult
    {
        double cost;
        bool has_fallback;
        vector<Path> paths; // one per agent
        int failed_calls;
    };

    std::clock_t start = 0;
    vector< Path* > paths;
    vector<Path*> best_paths;
    list< pair<int, Path> > paths_list;

    // PBSNode* dummy_start = nullptr;
    // vector<int> best_order;
    // std::vector<vector<int>> total_orders;
    bool find_path();
    // vector<int> select_best_order();

    double find_path_per_order(const std::vector<int>& total_order, bool fake_order, bool &order_has_fallback);

    OrderEvalResult evaluate_order(const std::vector<int>& total_order);

    // double find_path_per_order_local(const vector<int>& total_order, bool fake_order, bool &order_has_fallback,
    //     ReservationTable &local_rt, vector<Path*> &local_paths,
    //     vector<pair<int, Path>> &local_paths_list, SingleAgentSolver &local_planner);

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