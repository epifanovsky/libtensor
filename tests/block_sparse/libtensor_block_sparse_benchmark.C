/*
 * libtensor_block_sparse_benchmark.C
 *
 *  Created on: Nov 19, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/block_sparse/contract.h>
#include <libtensor/core/sequence.h>
#include <libtensor/iface/letter.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

using namespace libtensor;
using namespace std;

extern bool libtensor::count_flops;
extern size_t libtensor::flops;

double read_timer()
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }

    gettimeofday( &end, NULL );

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//Benchmark file format:
//N
//X
//
//all shell pair sparsity entries...
//
//all shell->aux atom sparsity entries...
void run_benchmark(const char* file_name)
{
    srand(time(NULL));

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

    //Construct C
    sparse_bispace<3> spb_C_orig = spb_N % spb_N % spb_X << sig_blocks_NNX;
    permutation<3> perm;
    perm.permute(1,2);
    sparse_bispace<3> spb_C = spb_C_orig.permute(perm);
    size_t nnz_C = spb_C.get_nnz();
    double* C_arr = new double[nnz_C];
    for(size_t i = 0; i < nnz_C; ++i)
    {
        C_arr[i] = double(rand())/RAND_MAX;
    }
    sparse_btensor<3> C(spb_C,C_arr,true);
    delete C_arr;

    //Construct P
    sparse_bispace<2> spb_P = spb_N|spb_N;
    size_t nnz_P = spb_P.get_nnz();
    double* P_arr = new double[nnz_P];
    for(size_t i = 0; i < nnz_P; ++i)
    {
        P_arr[i] = double(rand())/RAND_MAX;
    }
    sparse_btensor<2> P(spb_P,P_arr);
    delete P_arr;

    //Construct V
    sparse_btensor<2> V(spb_X|spb_X);
    for(size_t i = 0; i < (spb_X.get_nnz()*spb_X.get_nnz()); ++i)
    {
        //We know what we're doing - cast away const
        ((double*)V.get_data_ptr())[i] = double(rand())/RAND_MAX;
    }


    cout << "===========================\n";
    cout << "IN-CORE BENCHMARK:\n";

    //Construct D result
    sparse_bispace<3> spb_D = spb_C.contract(2) | spb_N;
    sparse_btensor<3> D(spb_D);
    letter mu,Q,lambda,sigma;
    cout << "-----------------------------\n";
    cout << "D(mu|Q|sigma) = contract(lambda,C(mu|Q|lambda),P(sigma|lambda))\n";
    flops = 0;
    count_flops = true;
    double seconds = read_timer();
    D(mu|Q|sigma) = contract(lambda,C(mu|Q|lambda),P(sigma|lambda));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    //Construct E tensor 
    sparse_btensor<3> C_aux_fast(spb_C_orig);
    letter nu,R;
    C_aux_fast(nu|sigma|Q) = C(nu|Q|sigma);
    sparse_bispace<3> spb_E = spb_X | spb_C.contract(1);
    sparse_btensor<3> E(spb_E);
    cout << "-----------------------------\n";
    cout << "E(Q|sigma|nu) = contract(R,V(Q|R),C_aux_fast(sigma|nu|R))\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    E(Q|sigma|nu) = contract(R,V(Q|R),C_aux_fast(sigma|nu|R));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";


    //Construct I - mock integral tensor
    sparse_btensor<3> I(spb_E);
    for(size_t i = 0; i < spb_E.get_nnz(); ++i)
    {
        //We know what we're doing - cast away const
        ((double*)I.get_data_ptr())[i] = double(rand())/RAND_MAX;
    }

    //Construct the L tensor
    cout << "-----------------------------\n";
    cout << "L(Q|sigma|nu) = I(Q|sigma|nu) - E(Q|sigma|nu)\n";
    sparse_btensor<3> L(spb_E);
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    L(Q|sigma|nu) = I(Q|sigma|nu) - E(Q|sigma|nu);
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    //Construct M result
    sparse_bispace<2> spb_M = spb_N|spb_N;
    sparse_btensor<2> M(spb_M);
    cout << "-----------------------------\n";
    cout << "M(nu|mu) = contract(Q|sigma,L(Q|sigma|nu),D(mu|Q|sigma))\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    M(nu|mu) = contract(Q|sigma,L(Q|sigma|nu),D(mu|Q|sigma));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    cout << "===========================\n";
    cout << "DIRECT BENCHMARK:\n";
    direct_sparse_btensor<3> D_direct(spb_D);
    direct_sparse_btensor<3> E_direct(spb_E);
    direct_sparse_btensor<3> L_direct(spb_E);
    D_direct(mu|Q|sigma) = contract(lambda,C(mu|Q|lambda),P(sigma|lambda));
    E_direct(Q|sigma|nu) = contract(R,V(Q|R),C_aux_fast(sigma|nu|R));
    L_direct(Q|sigma|nu) = I(Q|sigma|nu) - E_direct(Q|sigma|nu);

    sparse_btensor<2> M_from_direct(spb_M);
    cout << "-----------------------------\n";
    cout << "M(nu|mu) = contract(Q|sigma,E_direct(Q|sigma|nu),D_direct(mu|Q|sigma),400e6)\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    M_from_direct(nu|mu) = contract(Q|sigma,L_direct(Q|sigma|nu),D_direct(mu|Q|sigma),400e6);
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    cout << "===========================\n";
    cout << "Direct and Indirect Results Equal?\n";
    bool M_equal = true;
    for(size_t i = 0; i < spb_M.get_nnz(); ++i)
    {
        if(fabs(M_from_direct.get_data_ptr()[i] - M.get_data_ptr()[i]) > 0)
        {
            cout << "----------------\n";
            cout << "FAILURE!\n";
            cout << "Element idx: " << i << " out of " << spb_M.get_nnz() << "\n";
            cout << "Indirect: " << M.get_data_ptr()[i] << "\n";
            cout << "Direct: " << M_from_direct.get_data_ptr()[i] << "\n";
            cout << "Delta: " << fabs(M_from_direct.get_data_ptr()[i] - M.get_data_ptr()[i]) << "\n";
            M_equal = false;
            break;
        }
    }
    cout << "M_equal: " << (M_equal ? "YES" : "NO") << "\n";
}

int main(int argc,char *argv[])
{
    const char* alkane_file_names[7] = {"../tests/block_sparse/alkane_dz_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_atom_blocked_020_data.txt",
                                        "../tests/block_sparse/alkane_aTZ_ithrsh_10_010_data.txt",
                                        "../tests/block_sparse/alkane_aTZ_ithrsh_10_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/alkane_aTZ_ithrsh_14_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/anthracene_dz_atom_blocked.txt"};

    if(argc != 2)
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
        run_benchmark(alkane_file_names[benchmark_idx]);
    }
}
