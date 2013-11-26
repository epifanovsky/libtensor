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
#include <stdlib.h>
#include <sys/time.h>

using namespace libtensor;

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

void benchmark(std::vector<size_t>& split_points,std::vector< sequence<2,size_t> >& sig_blocks,size_t N)
{
    sparse_bispace<1> m(N);
	m.split(split_points);
    sparse_bispace<3> mnl = m % m << sig_blocks | m;
    
	//Don't want overflows
	double* A_arr  = new double[mnl.get_nnz()];
	double* B_arr  = new double[mnl.get_nnz()];
	for(size_t i = 0; i < mnl.get_nnz(); ++i)
	{
		A_arr[i] = double(rand())/RAND_MAX;
		B_arr[i] = double(rand())/RAND_MAX;
	}
    
	sparse_btensor<3> A(mnl,A_arr,true);
	sparse_btensor<3> B(mnl,B_arr,true);
	sparse_btensor<2> C(m|m);
    
	letter mu,nu,l,s;
	std::cout << "Flops before: " << flops << "\n";
	count_flops = true;
	double seconds = read_timer();
	C(mu|s) = contract(nu|l,A(mu|nu|l),B(nu|l|s));
	seconds = read_timer() - seconds;
	count_flops = false;
	std::cout << "Flops after: " << flops << "\n";
	std::cout << "MFLOPS/S: " << flops/(1e6*seconds);
}

void benchmark_graphene_01_03()
{
	//This test simulates an electronic structure calculation with NB2 sparsity
	//Data comes from a 1x3 graphene (anthracene) chain
    
	//Split the basis function space into atomic shells
	std::vector<size_t> split_points;
	std::ifstream split_ifstr;
	split_ifstr.open("../tests/block_sparse/graphene_01_03_split_points.txt");
	while(split_ifstr.good())
	{
		size_t val;
		split_ifstr >> val;
		split_points.push_back(val);
	}
    
	//Load the NB2 sparsity information
	std::vector< sequence<2,size_t> > sig_blocks;
	std::ifstream sb_ifstr;
	sb_ifstr.open("../tests/block_sparse/graphene_01_03_sig_blocks.txt");
	while(sb_ifstr.good())
	{
		sequence<2,size_t> entry;
		sb_ifstr >> entry[0] >> entry[1];
		sig_blocks.push_back(entry);
	}
    
    std::cout << "Benchmark running: graphene_01_03\n";
    benchmark(split_points,sig_blocks,246);
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

int main(int argc,char *argv[])
{

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
            benchmark_graphene_01_06();
        }
        else
        {
            std::cout << "Invalid benchmark specified!\n";
            print_avail_benchmarks();
        }
    }
}
