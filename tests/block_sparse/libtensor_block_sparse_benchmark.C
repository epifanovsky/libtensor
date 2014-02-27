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
#include <libtensor/expr/iface/letter.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <limits>
#include <string.h>

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
    perm.permute(0,2);
    sparse_bispace<3> spb_C = spb_C_orig.permute(perm);
    size_t nnz_C = spb_C.get_nnz();
    double* C_arr = new double[nnz_C];
    for(size_t i = 0; i < nnz_C; ++i)
    {
        C_arr[i] = double(rand())/RAND_MAX;
    }
    sparse_btensor<3> C(spb_C,C_arr,true);
    delete [] C_arr;

    //Construct P
    sparse_bispace<2> spb_P = spb_N|spb_N;
    size_t nnz_P = spb_P.get_nnz();
    double* P_arr = new double[nnz_P];
    for(size_t i = 0; i < nnz_P; ++i)
    {
        P_arr[i] = double(rand())/RAND_MAX;
    }
    sparse_btensor<2> P(spb_P,P_arr);
    delete [] P_arr;

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
    cout << "D(Q|mu|sigma) = contract(lambda,C(Q|mu|lambda),P(sigma|lambda))\n";
    flops = 0;
    count_flops = true;
    double seconds = read_timer();
    D(Q|mu|sigma) = contract(lambda,C(Q|mu|lambda),P(sigma|lambda));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    sparse_bispace<3> spb_D_perm = spb_D.permute(permutation<3>().permute(0,1).permute(1,2));
    sparse_btensor<3> D_perm(spb_D_perm);
    D_perm(mu|sigma|Q) = D(Q|mu|sigma);

    //Construct E tensor 
    sparse_btensor<3> C_aux_fast(spb_C_orig);
    letter nu,R;
    C_aux_fast(nu|sigma|Q) = C(Q|nu|sigma);
    sparse_bispace<3> spb_E =  spb_C_orig.contract(2) | spb_X;
    sparse_btensor<3> E(spb_E);
    cout << "-----------------------------\n";
    cout << "E(nu|sigma|Q) = contract(R,C_aux_fast(nu|sigma|R),V(Q|R))\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    E(nu|sigma|Q) = contract(R,C_aux_fast(nu|sigma|R),V(Q|R));
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
    cout << "L(nu|sigma|Q) = I(nu|sigma|Q) - E(nu|sigma|Q)\n";
    sparse_btensor<3> L(spb_E);
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    L(nu|sigma|Q) = I(nu|sigma|Q) - E(nu|sigma|Q);
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    //Construct M result
    sparse_bispace<2> spb_M = spb_N|spb_N;
    sparse_btensor<2> M(spb_M);

    //TODO: DEBUG REMOVE
    cout << "-----------------------------\n";
    cout << "M(nu|mu) = contract(sigma|Q,L(nu|sigma|Q),D_perm(mu|sigma|Q))\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    M(nu|mu) = contract(sigma|Q,L(nu|sigma|Q),D_perm(mu|sigma|Q));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    cout << "===========================\n";
    cout << "DIRECT BENCHMARK:\n";
    direct_sparse_btensor<3> D_direct(spb_D);
    direct_sparse_btensor<3> D_direct_perm(spb_D_perm);
    direct_sparse_btensor<3> E_direct(spb_E);
    direct_sparse_btensor<3> L_direct(spb_E);
    D_direct(Q|mu|sigma) = contract(lambda,C(Q|mu|lambda),P(sigma|lambda));
    D_direct_perm(mu|sigma|Q) = D_direct(Q|mu|sigma);

    //TODO: DEBUG REMOVE
    sparse_btensor<3> D_perm_from_direct(spb_D_perm);
    flops = 0;
    count_flops = true;
    D_perm_from_direct(mu|sigma|Q) = D_direct_perm(mu|sigma|Q);
    count_flops = false;

    E_direct(nu|sigma|Q) = contract(R,C_aux_fast(nu|sigma|R),V(Q|R));
    L_direct(nu|sigma|Q) = I(nu|sigma|Q) - E_direct(nu|sigma|Q);

    sparse_btensor<2> M_from_direct(spb_M);
    cout << "-----------------------------\n";
    cout << "M_from_direct(nu|mu) = contract(sigma|Q,L_direct(nu|sigma|Q),D_direct_perm(mu|sigma|Q),4e8)\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    M_from_direct(nu|mu) = contract(sigma|Q,L_direct(nu|sigma|Q),D_direct_perm(mu|sigma|Q),4e8);
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    cout << "===========================\n";
    cout << "Direct and Indirect Results Equal?\n";
    bool M_equal = true;
    //We use a loose error bound to account for intermediate roundoff errors
    sparse_btensor<2> error_mat(spb_M);
    //L is the only tensor not positive definite
    sparse_btensor<3> L_abs(spb_E);
    for(size_t i = 0; i < spb_E.get_nnz(); ++i)
    {
        double L_abs_val = fabs(L.get_data_ptr()[i]);
        double err_fac = std::numeric_limits<double>::epsilon()*spb_X.get_nnz()*spb_N.get_nnz();
        ((double*)L_abs.get_data_ptr())[i] = err_fac*L_abs_val;
    }
    error_mat(nu|mu) = contract(sigma|Q,L_abs(nu|sigma|Q),D_perm(mu|sigma|Q));

    for(size_t i = 0; i < spb_M.get_nnz(); ++i)
    {
        if(fabs(M_from_direct.get_data_ptr()[i] - M.get_data_ptr()[i]) > error_mat.get_data_ptr()[i])
        {
            cout << "----------------\n";
            cout << "FAILURE!\n";
            cout << "Element idx: " << i << " out of " << spb_M.get_nnz() << "\n";
            cout << "Indirect: " << M.get_data_ptr()[i] << "\n";
            cout << "Direct: " << M_from_direct.get_data_ptr()[i] << "\n";
            cout << "Delta: " << fabs(M_from_direct.get_data_ptr()[i] - M.get_data_ptr()[i]) << "\n";
            cout << "Allowed: " << error_mat.get_data_ptr()[i] << "\n";
            M_equal = false;
            break;
        }
    }
    cout << "M_equal: " << (M_equal ? "YES" : "NO") << "\n";
}

void run_benchmark_mo(const char* file_name)
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

    //Get the number of occupied orbitals
    getline(bif,line);
    size_t o = atoi(line.c_str());

    sparse_bispace<1> spb_N(N);
    spb_N.split(split_points_N);
    sparse_bispace<1> spb_X(X);
    spb_X.split(split_points_X);
    sparse_bispace<1> spb_o(o);

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
    delete [] C_arr;

    //Construct MO coeffs
    sparse_bispace<2> spb_mo = spb_N|spb_o;
    size_t nnz_mo = spb_mo.get_nnz();
    double* mo_arr = new double[nnz_mo];
    for(size_t i = 0; i < nnz_mo; ++i)
    {
        mo_arr[i] = double(rand())/RAND_MAX;
    }
    sparse_btensor<2> C_mo(spb_mo,mo_arr);
    delete [] mo_arr;

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
    sparse_bispace<3> spb_D = spb_C.contract(2) | spb_o;
    sparse_btensor<3> D(spb_D);
    letter mu,Q,lambda,i;
    cout << "-----------------------------\n";
    cout << "D(mu|Q|i) = contract(lambda,C(mu|Q|lambda),C_mo(lambda|i))\n";
    flops = 0;
    count_flops = true;
    double seconds = read_timer();
    D(mu|Q|i) = contract(lambda,C(mu|Q|lambda),C_mo(lambda|i));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    //Construct E tensor 
    sparse_btensor<3> C_aux_fast(spb_C_orig);
    letter nu,R,sigma;
    C_aux_fast(nu|sigma|Q) = C(nu|Q|sigma);
    sparse_bispace<3> spb_E =  spb_C_orig.contract(2) | spb_X;
    sparse_btensor<3> E(spb_E);
    cout << "-----------------------------\n";
    cout << "E(nu|sigma|Q) = contract(R,C_aux_fast(nu|sigma|R),V(Q|R))\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    E(nu|sigma|Q) = contract(R,C_aux_fast(nu|sigma|R),V(Q|R));
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

    //Construct the G tensor
    cout << "-----------------------------\n";
    cout << "G(nu|sigma|Q) = I(nu|sigma|Q) - E(nu|sigma|Q)\n";
    sparse_btensor<3> G(spb_E);
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    G(nu|sigma|Q) = I(nu|sigma|Q) - E(nu|sigma|Q);
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    //Permute G for MO-transform
    cout << "-----------------------------\n";
    cout << "G_perm(nu|Q|sigma) = G(nu|sigma|Q)\n";
    sparse_bispace<3> spb_G_perm = spb_E.permute(permutation<3>().permute(1,2));
    sparse_btensor<3> G_perm(spb_G_perm);
    seconds = read_timer();
    G_perm(nu|Q|sigma) = G(nu|sigma|Q);
    seconds = read_timer() - seconds;
    std::cout << "Time (s): " << seconds << "\n";

    //Construct the MO-transformed integrals
    cout << "-----------------------------\n";
    cout << "H(nu|Q|i) = contract(sigma,G_perm(nu|Q|sigma),C_mo(sigma|i))\n";
    sparse_bispace<3> spb_H = spb_G_perm.contract(2) | spb_o;
    sparse_btensor<3> H(spb_H);
    count_flops = true;
    seconds = read_timer();
    H(nu|Q|i) = contract(sigma,G_perm(nu|Q|sigma),C_mo(sigma|i));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    //Construct M result
    sparse_bispace<2> spb_M = spb_N|spb_N;
    sparse_btensor<2> M(spb_M);
    cout << "-----------------------------\n";
    cout << "M(nu|mu) = contract(Q|i,D(mu|Q|i),H(nu|Q|i))\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    M(mu|nu) = contract(Q|i,D(mu|Q|i),H(nu|Q|i));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    cout << "===========================\n";
    cout << "DIRECT BENCHMARK:\n";
    direct_sparse_btensor<3> D_direct(spb_D);
    direct_sparse_btensor<3> E_direct(spb_E);
    direct_sparse_btensor<3> G_direct(spb_E);
    direct_sparse_btensor<3> G_perm_direct(spb_G_perm);
    direct_sparse_btensor<3> H_direct(spb_H);

    D_direct(mu|Q|i) = contract(lambda,C(mu|Q|lambda),C_mo(lambda|i));
    E_direct(nu|sigma|Q) = contract(R,C_aux_fast(nu|sigma|R),V(Q|R));
    G_direct(nu|sigma|Q) = I(nu|sigma|Q) - E_direct(nu|sigma|Q);
    G_perm_direct(nu|Q|sigma) = G_direct(nu|sigma|Q);
    H_direct(nu|Q|i) = contract(sigma,G_perm_direct(nu|Q|sigma),C_mo(sigma|i));

    sparse_btensor<2> M_from_direct(spb_M);
    cout << "-----------------------------\n";
    cout << "M_from_direct(nu|mu) = contract(sigma|Q,L_direct(nu|sigma|Q),D_direct_perm(mu|sigma|Q),4e8)\n";
    flops = 0;
    count_flops = true;
    seconds = read_timer();
    M_from_direct(mu|nu) = contract(Q|i,D_direct(mu|Q|i),H_direct(nu|Q|i));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";

    cout << "===========================\n";
    cout << "Direct and Indirect Results Equal?\n";
    bool M_equal = true;
    //We use a loose error bound to account for intermediate roundoff errors
    //G is the only tensor not positive definite
    sparse_btensor<3> G_perm_abs(spb_G_perm);
    for(size_t i = 0; i < spb_G_perm.get_nnz(); ++i)
    {
        ((double*)G_perm_abs.get_data_ptr())[i] = fabs(G_perm_abs.get_data_ptr()[i]);
    }

    sparse_btensor<3> H_abs(spb_H);
    H_abs(nu|Q|i) = contract(sigma,G_perm_direct(nu|Q|sigma),C_mo(sigma|i));
    //Scale H by the error factor corresponding to doing the multiplation dense as a little hedge
    for(size_t i = 0; i < spb_H.get_nnz(); ++i)
    {
        double err_fac = std::numeric_limits<double>::epsilon()*spb_X.get_nnz()*spb_o.get_nnz();
        ((double*)H_abs.get_data_ptr())[i] = err_fac*((double*)H_abs.get_data_ptr())[i];
    }
    sparse_btensor<2> error_mat(spb_M);
    error_mat(mu|nu) = contract(Q|i,D(mu|Q|i),H_abs(nu|Q|i));

    for(size_t i = 0; i < spb_M.get_nnz(); ++i)
    {
        if(fabs(M_from_direct.get_data_ptr()[i] - M.get_data_ptr()[i]) > error_mat.get_data_ptr()[i])
        {
            cout << "----------------\n";
            cout << "FAILURE!\n";
            cout << "Element idx: " << i << " out of " << spb_M.get_nnz() << "\n";
            cout << "Indirect: " << M.get_data_ptr()[i] << "\n";
            cout << "Direct: " << M_from_direct.get_data_ptr()[i] << "\n";
            cout << "Delta: " << fabs(M_from_direct.get_data_ptr()[i] - M.get_data_ptr()[i]) << "\n";
            cout << "Allowed: " << error_mat.get_data_ptr()[i] << "\n";
            M_equal = false;
            break;
        }
    }
    cout << "M_equal: " << (M_equal ? "YES" : "NO") << "\n";
}

int main(int argc,char *argv[])
{
    const char* alkane_file_names[8] = {"../tests/block_sparse/alkane_dz_003_data.txt",
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

        size_t benchmark_type = 0;
        if(argc == 3)
        {
            if(strcmp(argv[2],"ao") == 0)
            {
                benchmark_type = 0;
            }
            else if(strcmp(argv[2],"mo") == 0)
            {
                benchmark_type = 1;
            }
            else
            {
                cout << "INVALID BENCHMARK TYPE!\n";
                cout << "Choose 'ao' or 'mo'\n";
                exit(1);
            }
        }

        if(benchmark_type == 0)
        {
            run_benchmark(alkane_file_names[benchmark_idx]);
        }
        else if(benchmark_type == 1)
        {
            run_benchmark_mo(alkane_file_names[benchmark_idx]);
        }
    }
}
