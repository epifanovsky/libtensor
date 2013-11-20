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

int main()
{
	//This test simulates an electronic structure calculation with NB2 sparsity
	//Data comes from a 1x3 graphene (anthracene) chain

	//Split the basis function space into atomic shells
	std::vector<size_t> split_points;
	std::ifstream split_ifstr;
	split_ifstr.open("../tests/block_sparse/split_points.txt");
	while(split_ifstr.good())
	{
		size_t val;
		split_ifstr >> val;
		split_points.push_back(val);
	}

	sparse_bispace<1> m(924);
	m.split(split_points);

	//Load the NB2 sparsity information
	std::vector< sequence<2,size_t> > sig_blocks;
	std::ifstream sb_ifstr;
	sb_ifstr.open("../tests/block_sparse/sig_blocks.txt");
	while(sb_ifstr.good())
	{
		sequence<2,size_t> entry;
		sb_ifstr >> entry[0] >> entry[1];
		sig_blocks.push_back(entry);
	}

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
	return 0;
}
