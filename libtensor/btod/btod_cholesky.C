#include <iostream>
#include "../core/allocator.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/block_tensor.h"
#include "btod_add.h"
#include "btod_contract2.h"
#include "btod_copy.h"
#include "btod_diag.h"
#include "btod_extract.h"
#include "btod_scale.h"
#include "btod_set_diag.h"
#include "btod_set_elem.h"
#include "btod_set.h"
#include "btod_cholesky.h"
#include "btod_import_raw.h"
#include <libtensor/linalg.h> //necessary to include LAPACK functions

#include "../core/tensor.h"
#include "../core/tensor_ctrl.h"
#include "../tod/tod_btconv.h"


//#define PRINT 1


namespace libtensor
{
btod_cholesky::btod_cholesky(block_tensor_i<2, double> &bta, double tol) :
	m_bta(bta), m_tol(tol){

}

void btod_cholesky::perform(block_tensor_i<2, double> &btb)
	{
        
	/*
	size_t n = 5, n2 = n * n;
        
        double *mat = new double [n2];

        mat[0] = 1; mat[1] = 0; mat[2] = 0; mat[3] = 0; mat[4] = 0;
        mat[5] = 1; mat[6] = 2; mat[7] = 0; mat[8] = 0; mat[9] = 0; 
	mat[10] = 1; mat[11] = 3; mat[12] = 6; mat[13] = 0; mat[14] = 0;
	mat[15] = 1; mat[16] = 4; mat[17] = 10; mat[18] = 20; mat[19] = 0;
	mat[20] = 1; mat[21] = 5; mat[22] = 15; mat[23] = 35; mat[24] = 70;
	*/
       
	// form the tensor with initial data
        typedef std_allocator<double> allocator_t;
        const dimensions<2> &dims = btb.get_bis().get_dims();
	//create the tensor
	tensor<2, double, allocator_t> ta(dims);
	//fill tensor with data from source btensor
        tod_btconv<2>(m_bta).perform(ta);

	tensor_ctrl<2, double> tnsr_ctrl(ta);
        double *tnsr_ptr = tnsr_ctrl.req_dataptr();

	size_t n = dims.get_dim(0);// size of the matrix

	// initialize the workspace
	int *p = new int [n];//PIV
	int *rank = new int; //rank
	double *work = new double[2 * n];
	
	*rank = 0;	
	for(size_t k = 0; k < n; k++)
	{
	*(p+k) = 0;
	}
	for(size_t k = 0; k < 2 * n; k++)
	{
	*(work+k) = 0;
	}

	#ifdef PRINT
	std::cout<<std::endl;
	std::cout<<"Parameters before solver are:"<<std::endl; 
	std::cout<<"Tolerance is "<<m_tol<<std::endl;
	std::cout<<"Size of the matrix is "<<n<<std::endl;
	#endif

	if(libtensor::lapack_dpstrf('U', n, tnsr_ptr , n, p, rank, m_tol, work) != 0) {
                throw 1;
                //      exception: failure to decompose matrix
        }

	//make zeros above the diagonal

	for(size_t i =0 ; i < n; i++)
        {
                for(size_t j = i + 1 ; j < n; j++)
                {
                        *(tnsr_ptr + j + i * n) = 0;
                }
        }
 	
	#ifdef PRINT
	std::cout<<"Parameters after solver "<<std::endl;
	std::cout<<"PIV"<<std::endl;

	for(int i = 0; i < n; i++)
	{
	std::cout<<*(p+i)<<std::endl;
	}
	
	std::cout<<"Rank is "<<*rank<<std::endl;
	#endif
	
	// create permutation matrix
	double perm[n * n];
	bool permzero = 1;

	for(int i = 0; i < n; i ++)
	{
		for (int j=0; j<n; j++)
		{
		if(*(p+j) == i + 1)
		{
		perm[i * n + j] = 1;
		permzero = 0;
		}
		else
		{
		perm[i * n + j] = 0;
		}

		}
	}

	//print perm
	/*
	for(int i = 0; i < n; i ++)
        {
		std::cout<<std::endl;
                for (int j=0; j<n; j++)
                {
                
		std::cout<<perm[i * n + j];

                }
        }
	*/
	// temp matrix

        //put data to the output block tensor
        btod_import_raw<2>(tnsr_ptr, dims).perform(btb);
	tnsr_ctrl.ret_dataptr(tnsr_ptr);

	if(!permzero)
	{
	block_tensor<2, double, allocator_t> tmp(btb.get_bis());
        btod_import_raw<2>(perm, dims).perform(tmp);
        // now tmp has permutation matrix

        //second tmp matrix
        block_tensor<2, double, allocator_t> tmp2(btb.get_bis());
        btod_copy<2>(btb).perform(tmp2);

        //now tmp2 has btb
	contraction2<1,1,1> contr;
        contr.contract(1,0);
        btod_contract2<1,1,1>(contr,tmp,tmp2).perform(btb);
	}
	// cleanup

	delete[] p;
	delete rank;
	delete[] work;

}


}//namespace libtensor
