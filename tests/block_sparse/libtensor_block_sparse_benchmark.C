/*
 * libtensor_block_sparse_benchmark.C
 *
 *  Created on: Nov 19, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/contract.h>
#include <libtensor/core/sequence.h>
#include <libtensor/iface/letter.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>

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

#if 0
void benchmark(std::vector<size_t>& split_points_N,
               std::vector<size_t>& split_points_X, 
               std::vector< sequence<2,size_t> >& sig_blocks_NN,
               std::vector< sequence<2,size_t> >& sig_blocks_NX,
               size_t N,
               size_t X)
{
    /*for(size_t i = 0; i < split_points_N.size(); ++i)*/
    /*{*/
        /*std::cout << split_points_N[i] << "\n";*/
    /*}*/

    /*std::cout << "split_points_X:\n";*/
    /*for(size_t i = 0; i < split_points_X.size(); ++i)*/
    /*{*/
        /*std::cout << split_points_X[i] << "\n";*/
    /*}*/

    /*std::cout << "sig_blocks_NN:\n";*/
    /*for(size_t i = 0; i < sig_blocks_NN.size(); ++i)*/
    /*{*/
        /*std::cout << sig_blocks_NN[i][0] << "," << sig_blocks_NN[i][1] << "\n";*/
    /*}*/

    /*std::cout << "sig_blocks_NX:\n";*/
    /*for(size_t i = 0; i < sig_blocks_NX.size(); ++i)*/
    /*{*/
        /*std::cout << sig_blocks_NX[i][0] << "," << sig_blocks_NX[i][1] << "\n";*/
    /*}*/
    

    //Shell pair sparsity test
    {
    sparse_bispace<1> m(N);
	m.split(split_points_N);
    sparse_bispace<3> spb_A = m % m << sig_blocks_NN | m;
    sparse_bispace<3> spb_B = m | m % m << sig_blocks_NN;
    
    //Don't want overflows
    double* A_arr  = new double[spb_A.get_nnz()];
    double* B_arr  = new double[spb_B.get_nnz()];
    for(size_t i = 0; i < spb_A.get_nnz(); ++i)
    {
        A_arr[i] = double(rand())/RAND_MAX;
    }
    for(size_t i = 0; i < spb_B.get_nnz(); ++i)
    {
        B_arr[i] = double(rand())/RAND_MAX;
    }
    
    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<3> B(spb_B,B_arr,true);
    sparse_btensor<2> C(m|m);
    
    letter mu,nu,Q,s;
    std::cout << "==========================\n";
    std::cout << "SMALL BLOCK BENCHMARK: ONLY NB2 SPARSITY\n";
    std::cout << "Flops before: " << flops << "\n";
    count_flops = true;
    double seconds = read_timer();
    C(mu|nu) = contract(Q|s,A(mu|Q|s),B(Q|s|nu));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "Flops after: " << flops << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";
    }

    //shell->aux atom pair sparsity test
    {
    sparse_bispace<1> spb_N(N);
    spb_N.split(split_points_N);
    sparse_bispace<1> spb_X(X);
    spb_X.split(split_points_X);

    sparse_bispace<3> spb_A = spb_N % spb_X << sig_blocks_NX | spb_N;
    sparse_bispace<3> spb_B = spb_X | spb_N % spb_N << sig_blocks_NN;
    
    //Don't want overflows
    double* A_arr  = new double[spb_A.get_nnz()];
    double* B_arr  = new double[spb_B.get_nnz()];
    for(size_t i = 0; i < spb_A.get_nnz(); ++i)
    {
        A_arr[i] = double(rand())/RAND_MAX;
    }
    for(size_t i = 0; i < spb_B.get_nnz(); ++i)
    {
        B_arr[i] = double(rand())/RAND_MAX;
    }
    
    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<3> B(spb_B,B_arr,true);
    sparse_btensor<2> C(spb_N|spb_N);
    
    letter mu,nu,Q,s;
    flops = 0;
    std::cout << "==========================\n";
    std::cout << "LARGE BLOCK BENCHMARK: SHELL/AUX ATOM SPARSITY\n";
    std::cout << "Flops before: " << flops << "\n";
    count_flops = true;
    double seconds = read_timer();
    C(mu|nu) = contract(Q|s,A(mu|Q|s),B(Q|s|nu));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "Flops after: " << flops << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";
    }
}

void benchmark_graphene_01_03()
{
	//This test simulates an electronic structure calculation with NB2 sparsity
	//Data comes from a 1x3 graphene (anthracene) chain - cc-pVDZ basis, rimp2-aug-cc-pVQZ aux basis
    
	//Split the basis function space into atomic shells
	std::vector<size_t> split_points_N;
	std::ifstream split_ifstr;
	split_ifstr.open("../tests/block_sparse/graphene_01_03_split_points_N.txt");
	while(split_ifstr.good())
	{
		size_t val;
		split_ifstr >> val;
        if(split_ifstr.eof())
        {
            break;
        }
		split_points_N.push_back(val);
	}

	//Split the aux basis into atom blocks 
	std::vector<size_t> split_points_X;
    std::ifstream split_ifstr_X("../tests/block_sparse/graphene_01_03_split_points_X.txt");
	while(split_ifstr_X.good())
	{
		size_t val;
		split_ifstr_X >> val;
        if(split_ifstr_X.eof())
        {
            break;
        }
		split_points_X.push_back(val);
	}
    
	//Load the NB2 sparsity information
	std::vector< sequence<2,size_t> > sig_blocks_NN;
	std::ifstream sb_ifstr_N("../tests/block_sparse/graphene_01_03_sig_blocks_NN.txt");
	while(sb_ifstr_N.good())
	{
		sequence<2,size_t> entry;
		sb_ifstr_N >> entry[0] >> entry[1];
        if(sb_ifstr_N.eof())
        {
            break;
        }
		sig_blocks_NN.push_back(entry);
	}

    //Load the basis shell->aux atom block sparsity information
	std::vector< sequence<2,size_t> > sig_blocks_NX;
    std::ifstream sb_ifstr_X("../tests/block_sparse/graphene_01_03_sig_blocks_NX.txt");
	while(sb_ifstr_X.good())
	{
		sequence<2,size_t> entry;
		sb_ifstr_X >> entry[0] >> entry[1];
        if(sb_ifstr_X.eof())
        {
            break;
        }
		sig_blocks_NX.push_back(entry);
	}
    
    std::cout << "Benchmark running: graphene_01_03\n";
    benchmark(split_points_N,split_points_X,sig_blocks_NN,sig_blocks_NX,246,3152);
}

void benchmark_graphene_01_06()
{
    //This test simulates an electronic structure calculation with NB2 sparsity
	//Data comes from a 1x6 graphene chain
    
	//Split the basis function space into atomic shells
	std::vector<size_t> split_points;
	std::ifstream split_ifstr;
	split_ifstr.open("../tests/block_sparse/graphene_01_06_split_points.txt");
	while(split_ifstr.good())
	{
		size_t val;
		split_ifstr >> val;
		split_points.push_back(val);
	}
    
	//Load the NB2 sparsity information
	std::vector< sequence<2,size_t> > sig_blocks;
	std::ifstream sb_ifstr;
	sb_ifstr.open("../tests/block_sparse/graphene_01_06_sig_blocks.txt");
	while(sb_ifstr.good())
	{
		sequence<2,size_t> entry;
		sb_ifstr >> entry[0] >> entry[1];
		sig_blocks.push_back(entry);
	}
    
    std::cout << "Benchmark running: graphene_01_06\n";
    benchmark(split_points,sig_blocks,444);
}

void print_avail_benchmarks()
{
    std::cout << "Available benchmarkes are:\n";
    std::cout << "\t" << "graphene_01_03\n";
    std::cout << "\t" << "graphene_01_06\n";
}
#endif

//Benchmark file format:
//N
//X
//
//all shell pair sparsity entries...
//
//all shell->aux atom sparsity entries...
void run_benchmark(const char* file_name)
{
    size_t N;
    size_t X;
    vector<size_t> split_points_N;
    vector<size_t> split_points_X;
    vector< sequence<2,size_t> > sig_blocks_NN;
    vector< sequence<2,size_t> > sig_blocks_NX;
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

    //Get the shell-aux atom sparsity information
    getline(bif,line);
    while(line.length() > 0)
    {
        sequence<2,size_t> entry;
        istringstream(line) >> entry[0] >> entry[1];
        sig_blocks_NX.push_back(entry);
        getline(bif,line);
    }

    sparse_bispace<1> spb_N(N);
    spb_N.split(split_points_N);
    sparse_bispace<1> spb_X(X);
    spb_X.split(split_points_X);

    sparse_bispace<3> spb_A = spb_N % spb_X << sig_blocks_NX | spb_N;
    sparse_bispace<3> spb_B = spb_X | spb_N % spb_N << sig_blocks_NN;
    
    //Don't want overflows
    size_t nnz_A = spb_A.get_nnz();
    size_t nnz_B = spb_B.get_nnz();
    double* A_arr  = new double[nnz_A];
    double* B_arr  = new double[nnz_B];
    for(size_t i = 0; i < nnz_A; ++i)
    {
        A_arr[i] = double(rand())/RAND_MAX;
    }
    for(size_t i = 0; i < nnz_B; ++i)
    {
        B_arr[i] = double(rand())/RAND_MAX;
    }
    
    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<3> B(spb_B,B_arr,true);
    sparse_btensor<2> C(spb_N|spb_N);
    
    letter mu,nu,Q,s;
    flops = 0;
    count_flops = true;
    double seconds = read_timer();
    std::cout << "Starting contraction:\n";
    C(mu|nu) = contract(Q|s,A(mu|Q|s),B(Q|s|nu));
    seconds = read_timer() - seconds;
    count_flops = false;
    std::cout << "FLOPs: " << flops << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "MFLOPS/S: " << flops/(1e6*seconds) << "\n";
}

int main(int argc,char *argv[])
{

#if 0
    //Default is to run short benchmark
    if(argc < 2)
    {
        benchmark_graphene_01_03();
    }
    else if(argc > 2)
    {
        std::cout << "Error: Specify only one benchmark to run at a time!\n";
        print_avail_benchmarks();
    }
    else
    {
        if(strcmp(argv[1],"graphene_01_03") == 0)
        {
            benchmark_graphene_01_03();
        }
        else if(strcmp(argv[1],"graphene_01_06") == 0)
        {
            /*benchmark_graphene_01_06();*/
        }
        else
        {
            std::cout << "Invalid benchmark specified!\n";
            print_avail_benchmarks();
        }
    }
#endif
    const char* alkane_file_names[5] = {"../tests/block_sparse/alkane_dz_atom_blocked_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_atom_blocked_020_data.txt",
                                        "../tests/block_sparse/alkane_dz_010_data.txt",
                                        "../tests/block_sparse/alkane_dz_020_data.txt",
                                        "../tests/block_sparse/alkane_tz_010_data.txt"
                                        };
    if(argc < 2)
    {
        for(size_t i = 0; i < sizeof(alkane_file_names)/sizeof(alkane_file_names[0]); ++i)
        {
            cout << "=================\n";
            cout << alkane_file_names[i] << "\n";
            run_benchmark(alkane_file_names[i]);
        }
    }
    else if(argc == 2)
    {
        size_t benchmark_idx = atoi(argv[1]);
        cout << "=================\n";
        cout << alkane_file_names[benchmark_idx] << "\n";
        run_benchmark(alkane_file_names[benchmark_idx]);
    }
}
