#include <iostream>
#include "../core/allocator.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/block_tensor.h"
#include "scalar_transf_double.h"
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

#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_import_raw.h>
#include <libtensor/dense_tensor/tod_copy.h>



//#define PRINT 1


namespace libtensor
{
btod_cholesky::btod_cholesky(block_tensor_i<2, double> &bta, double tol) :
    m_bta(bta), m_tol(tol),  
    pta(new dense_tensor<2, double, std_allocator <double> >(bta.get_bis().get_dims()))
    {

}

btod_cholesky::~btod_cholesky(){
    delete pta;
    pta = NULL;
}




void btod_cholesky::decompose()
    {
    
    dense_tensor_i<2, double> &ta(*pta);

    // put the data from input matrix to the buffer
    typedef std_allocator<double> allocator_t;
    const dimensions<2> &dims = m_bta.get_bis().get_dims();
    tod_btconv<2>(m_bta).perform(ta);
    
    dense_tensor_ctrl<2, double> tnsr_ctrl(ta);
        double *tnsr_ptr = tnsr_ctrl.req_dataptr();

    size_t n = dims.get_dim(0);// size of the matrix

    // initialize the workspace
    int *p = new int [n];//PIV
    int *rank; rank = &m_rank; *rank = n;
    double *work = new double[2 * n];
        
    for(size_t k = 0; k < n; k++)
    {
    *(p+k) = 0;
    }
    for(size_t k = 0; k < 2 * n; k++)
    {
    *(work+k) = 0;
    }


    #ifdef PRINT

    /*
    std::cout<<"The buffer before cholesky"<<std::endl;
    for(int i = 0; i < n; i++)
    {
        std::cout<<std::endl;
        for(int j = 0; j < n; j++)
        {
        std::cout<<tnsr_ptr[i*n + j]<<" ";
        }
    }
    */
    std::cout<<std::endl;
    std::cout<<"Parameters before solver are:"<<std::endl; 
    std::cout<<"Tolerance is "<<m_tol<<std::endl;
    std::cout<<"Size of the matrix is "<<n<<std::endl;
    std::cout<<"Rank is "<<*rank<<std::endl;
    #endif

    if(libtensor::lapack_dpstrf('U', n, tnsr_ptr , n, p, rank, m_tol, work) != 0) {
                throw 1;
                //      exception: failure to decompose matrix
        }

    //make zeros above the diagonal
    //!!!!!!!!!!!!!!!!!!!
    //are the limits correct? to n or to *rank? pretty sure to n
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
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // does it go to rank or to n? should go to n
    for(int i = 0; i < n; i++)
    {
    std::cout<<*(p+i)<<std::endl;
    }
    
    std::cout<<"Rank is "<<*rank<<std::endl;
    #endif
    
    // create permutation matrix
    double perm[n * n];
    bool permzero = 1;

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // I am not sure about limits
    // should it be n or *rank for j. Pretty sure should be n
    for(int i = 0; i < n; i ++)
    {
        for (int j=0; j < n; j++)
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
    #ifdef PRINT
    std::cout<<"Permutational matrix is "<<std::endl;
    for(int i = 0; i < n; i ++)
        {
        std::cout<<std::endl;
                for (int j=0; j<n; j++)
                {
                
        std::cout<<perm[i * n + j];

                }
        }
    #endif

        tnsr_ctrl.ret_dataptr(tnsr_ptr);
    
    // if the permutational matrix is non-trivial => 
    // apply it to the buffer ta
    if(!permzero)
    {
        block_tensor<2, double, allocator_t> tmp(m_bta.get_bis());
        btod_import_raw<2>(perm, dims).perform(tmp);
        // now tmp has permutation matrix

        //second tmp matrix
    double *tnsr_ptr2 = tnsr_ctrl.req_dataptr();
        block_tensor<2, double, allocator_t> tmp2(m_bta.get_bis());
        btod_import_raw<2>(tnsr_ptr2,dims).perform(tmp2);
    tnsr_ctrl.ret_dataptr(tnsr_ptr2);
    // noew second tmp matrix has the information form the buffer

    // tmp for the new buffer
        block_tensor<2, double, allocator_t> tmp3(m_bta.get_bis());

        //apply permutation
        contraction2<1,1,1> contr;
        contr.contract(1,0);
        btod_contract2<1,1,1>(contr,tmp,tmp2).perform(tmp3);
    
    // update buffer
        tod_btconv<2>(tmp3).perform(ta);
    }
    // cleanup
    
    delete[] p;
    delete[] work;

}


void btod_cholesky::perform(block_tensor_i<2 , double> &btb)
{
        dense_tensor_i<2, double> &ta(*pta);
    dense_tensor_ctrl<2, double> tnsr_ctrl(ta);
        double *tnsr_ptr = tnsr_ctrl.req_dataptr();
    // temporary solution  - make the buffer of the size n by rank

    int n = btb.get_bis().get_dims().get_dim(0);
        int R = btb.get_bis().get_dims().get_dim(1);

    double tmp [n * R];

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < R; j++)
        {
            tmp[i * R + j] = *(tnsr_ptr + i * n + j);
        }
    } 
    //btod_import_raw<2>(tnsr_ptr, btb.get_bis().get_dims()).perform(btb);
    btod_import_raw<2>(tmp, btb.get_bis().get_dims()).perform(btb);
    tnsr_ctrl.ret_dataptr(tnsr_ptr);
}

}//namespace libtensor
