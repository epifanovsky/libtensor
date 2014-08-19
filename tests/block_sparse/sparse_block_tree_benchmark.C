#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/expr/operators/contract.h>
#include <libtensor/core/sequence.h>
#include <libtensor/expr/iface/letter.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <string.h>
#include "timer.h"

using namespace libtensor;
using namespace std;

void run_benchmark(const char* file_name)
{
    srand(time(NULL));
    double start_seconds = read_timer();

    size_t N;
    size_t X;
    vector<size_t> split_points_N;
    vector<size_t> split_points_X;
    vector< sequence<2,size_t> > sig_blocks_NN;
    vector< sequence<3,size_t> > sig_blocks_NNX;
    string line;
    ifstream bif(file_name);

    getline(bif,line);
    istringstream(line) >> N;
    getline(bif,line);
    istringstream(line) >> X;

    //Read N splitting information
    //Skip section header 
    getline(bif,line);
    getline(bif,line);
    while(line.length() > 0)
    {
        size_t entry;
        istringstream(line) >> entry;
        split_points_N.push_back(entry);
        getline(bif,line);
    }

    //Read the X splitting information  
    getline(bif,line);
    while(line.length() > 0)
    {
        size_t entry;
        istringstream(line) >> entry;
        split_points_X.push_back(entry);
        getline(bif,line);
    }

    //Get the shell pair sparsity information
    getline(bif,line);
    while(line.length() > 0)
    {
        sequence<2,size_t> entry;
        istringstream(line) >> entry[0] >> entry[1];
        sig_blocks_NN.push_back(entry);
        getline(bif,line);
    }

    //Get the shell-shell-aux atom sparsity information
    getline(bif,line);
    while(line.length() > 0)
    {
        sequence<3,size_t> entry;
        istringstream(line) >> entry[0] >> entry[1] >> entry[2];
        sig_blocks_NNX.push_back(entry);
        getline(bif,line);
    }

    sparse_bispace<1> spb_N(N);
    spb_N.split(split_points_N);
    sparse_bispace<1> spb_X(X);
    spb_X.split(split_points_X);

    cout << "===========================\n";
    cout << "PERMUTE BENCHMARK:\n";
    cout << "\tFully Sparse: ";
    sparse_bispace<3> spb_C_orig = spb_N % spb_N % spb_X << sig_blocks_NNX;
    permutation<3> perm;
    perm.permute(1,2);
    double seconds = read_timer();
    sparse_bispace<3> spb_C = spb_C_orig.permute(perm);
    cout << read_timer() - seconds << " s\n";

    sparse_bispace<3> spb_E =  spb_C_orig.contract(2) | spb_X;
    //Sparse tree is unbroken, just relocated
    permutation<3> E_perm_fast;
    E_perm_fast.permute(0,2);
    E_perm_fast.permute(1,2);
    cout << "\tPartially Sparse: ";
    seconds = read_timer();
    sparse_bispace<3> spb_E_perm_fast = spb_E.permute(E_perm_fast);
    cout << read_timer() - seconds << " s\n";

    permutation<3> E_perm_slow;
    E_perm_slow.permute(1,2);
    cout << "\tDense Incorporation: ";
    seconds = read_timer();
    sparse_bispace<3> spb_E_perm_slow = spb_E.permute(E_perm_slow);
    cout << read_timer() - seconds << " s\n";

    cout << "===========================\n";
    cout << "FUSE BENCHMARK:\n";
    sparse_block_tree E_perm_slow_tree = spb_E_perm_slow.get_sparse_group_tree(0);
    sparse_block_tree E_tree = spb_E.get_sparse_group_tree(0);

    // nu Q sigma fused to nu sigma Q, fusing nu and sigma
    idx_list lhs_fused_inds(2);
    lhs_fused_inds[0] = 0;
    lhs_fused_inds[1] = 2;
    idx_list rhs_fused_inds(2);
    rhs_fused_inds[0] = 0;
    rhs_fused_inds[1] = 1;
    cout << "\tfully fused: ";
    seconds = read_timer();
    sparse_block_tree fused = E_perm_slow_tree.fuse(E_tree,lhs_fused_inds,rhs_fused_inds);
    cout << read_timer() - seconds << " s\n";

    cout << "===========================\n";
    cout << "COPY BENCHMARK:\n";
    cout << "\tDirect copy: ";
    seconds = read_timer();
    sparse_block_tree copied_tree = E_tree;
    cout << read_timer() - seconds << " s\n";
    cout << "===========================\n";
    cout << "ITER BENCHMARK:\n";
    cout << "\tfused iter: ";
    seconds = read_timer();
    size_t count = 0;
    for(sparse_block_tree::iterator it = fused.begin(); it != fused.end(); ++it)
    {
        ++count;
    }
    cout << read_timer() - seconds << " s\n";
    cout << "TIME IN FUNCTION SCOPE: " << read_timer() - start_seconds << " s\n";
}

int main(int argc,char *argv[])
{
    const char* alkane_file_names[9] = {"../tests/block_sparse/alkane_dz_003_data.txt",
                                        "../tests/block_sparse/alkane_tz_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_atom_blocked_020_data.txt",
                                        "../tests/block_sparse/alkane_aTZ_ithrsh_10_010_data.txt",
                                        "../tests/block_sparse/alkane_aTZ_ithrsh_10_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/alkane_aTZ_ithrsh_14_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/anthracene_dz_atom_blocked.txt"};

    if(argc < 2)
    {
        cout << "Usage: [program name] [benchmark #]\n";
        cout << "Available benchmarks:\n";
        for(size_t  i = 0; i < sizeof(alkane_file_names)/sizeof(alkane_file_names[0]); ++i)
        {
            cout << i << ". " << alkane_file_names[i] << "\n";
        }
    }
    else
    {
        size_t benchmark_idx = atoi(argv[1]);
        cout << "=================\n";
        cout << alkane_file_names[benchmark_idx] << "\n";
        double seconds = read_timer();
        run_benchmark(alkane_file_names[benchmark_idx]);
        cout << "TOTAL TIME: " << read_timer() - seconds << "\n"; 
    }
}
