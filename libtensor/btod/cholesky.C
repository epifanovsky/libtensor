#include <iostream>
#include "../core/allocator.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/block_tensor.h"
#include "btod_add.h"
#include "btod_contract2.h"
#include "btod_copy.h"
#include "btod_diag.h"
#include "cholesky.h"
#include "btod_extract.h"
#include "btod_scale.h"
#include "btod_set_diag.h"
#include "btod_set_elem.h"
#include "btod_set.h"

#include "btod_print.h"

//#define PRINT


namespace libtensor
{

cholesky::cholesky(block_tensor_i<2, double> &bta, double tol,int maxiter) :
m_bta(bta) ,m_tol(tol),m_maxiter(maxiter),m_iter(0), doneiter(0), m_rank(0)
, pta(new block_tensor<2, double, std_allocator <double> >(bta.get_bis()))

{

}

cholesky::~cholesky(){
        delete pta;
        pta = NULL;
}


void cholesky::decompose()
{
	
	block_tensor_i<2, double> &buff(*pta);
	btod_copy<2>(m_bta).perform(buff);

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// zero out small elements
	/*
	// find highest diag element
	index<1> i1a, i1b;
        i1b[0] = buff.get_bis().get_dims().get_dim(0);

        dimensions<1> dims1(index_range<1>(i1a, i1b));

        block_index_space<1> bis1(dims1);

        block_tensor<1, double, allocator_t> diag(bis1);//input matrix

	mask<2> msk;
        msk[0] = true; msk[1] = true;
        btod_diag<2, 2>(buff, msk).perform(diag);

	//!!!!!!!!!!!!!!!!!!!!!!!
	// here I should run other all diag elements and fing highes one

	// should be divided by max element
	double thresh = m_tol;

	//run over the tensor and zero out everythin smaller than thresh

	*/
	
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// make permutations
	

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// find Cholesky
	size_t n = buff.get_bis().get_dims().get_dim(0);//size of the matrix

	index<2> idx;
        idx[1] = 0;

	size_t pos1 = 0;

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// change to the maxdiag from the prev routine when we zero out
	double maxdiag = 100;
	// CHANGE LIMITS FROM N-2 TO N
	for(size_t i1 = 0; i1 < n  && maxdiag > m_tol; i1++)
	{
	// first run over index 1 (columns of the matrices)
        if(buff.get_bis().get_block_dims(idx).get_dim(1) - 1 < pos1)
        {
                pos1 -= buff.get_bis().get_block_dims(idx).get_dim(1);
                idx[1]++;
        }

        idx[0] = 0;
        size_t pos0 = 0;
	
	for(size_t i0 = 0 ; i0 < n; i0++)
	{
	// after that run over index 0 (rows of the matrix)

                if(buff.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos0)
                {
                        pos0 -= buff.get_bis().get_block_dims(idx).get_dim(0);
                        idx[0]++;
                }

	block_tensor_ctrl<2, double> ctrl(buff);

        tensor_i<2, double> &t = ctrl.req_block(idx);
        tensor_ctrl<2, double> ct(t);
       	double *p = ct.req_dataptr();

	if( i0 < i1)//above diagonal
	{
	//make zeros

	size_t np = buff.get_bis().get_block_dims(idx).get_dim(1);

	*(p + pos0 * np + pos1) = 0;

	ct.ret_dataptr(p);
        ctrl.ret_block(idx);
	}
	else if(i0 == i1)// diagonal
	{
	// L_{ii} = sqrt(V_{ii});

	size_t np = buff.get_bis().get_block_dims(idx).get_dim(1);

	double tmp = *(p + pos0 * np + pos1);
	if (tmp < 0)
	{
	std::cout<<std::endl;
	std::cout<<"Error in Cholesky factorization - element "<<i0<<","<<i1<<" is less than zero ("<<tmp<<")"<<std::endl;
	throw 1;
	}

	*(p + pos0 * np + pos1) = sqrt(tmp);

        ct.ret_dataptr(p);
        ctrl.ret_block(idx);
	}
	else// non -diagonal
	{
	// 1. L_{ij} = V_{ij} - \sum_{k=1}^{i-1} L_{jk} L_{ik}/L_{ii}
	
	size_t np = buff.get_bis().get_block_dims(idx).get_dim(1);

	// tmp contains V_{ij}
	double tmp;
	tmp = *(p + pos0 * np + pos1);

        ct.ret_dataptr(p);
        ctrl.ret_block(idx);

        index<2> idxp;

        idxp[0] = idx[0];
        idxp[1] = 0 ;

        size_t posp0 = pos0;
        size_t posp1 = 0;

	double *pp;

	// subtract \sum_{k=1}^{i-1} L_{jk} L_{ik}
	if( i1 > 0)
	{
	
	for(size_t k = 0; k <= (i1 - 1); k++)
	{
	

	idxp[0] = idx[0];
	idxp[1] = 0 ;

	posp0 = pos0;
	posp1 = 0;

	// find posp1 and idxp[1]
	// This is for L_{jk}

	for(size_t m = 0; m < k; m++)
	{
	posp1++;
	if(buff.get_bis().get_block_dims(idxp).get_dim(1) - 1 < posp1)
	{
		posp1 -= buff.get_bis().get_block_dims(idxp).get_dim(1);
                idxp[1]++;
	}
	// should it be else here?
	}
	
	tensor_i<2 ,double> &tt = ctrl.req_block(idxp); 
	tensor_ctrl<2, double> ctt(tt);	
        pp = ctt.req_dataptr();

	size_t np = buff.get_bis().get_block_dims(idxp).get_dim(1);	
	double tmp1 = *(pp + posp0 * np + posp1);

	ctt.ret_dataptr(pp);
        ctrl.ret_block(idxp);

	idxp[0] = 0;
	posp0 = 0;

	//find posp0 and inxp[0]
	// FInd L_{ik}
	
	for(size_t m = 0; m < i1; m++)
	{
	posp0++;
        if(buff.get_bis().get_block_dims(idxp).get_dim(0) - 1 < posp0)
        {
                posp0 -= buff.get_bis().get_block_dims(idxp).get_dim(0);
                idxp[0]++;
        }
	}

	tensor_i<2 ,double> &t2 = ctrl.req_block(idxp); 
	tensor_ctrl<2, double> ct2(t2);
        pp = ct2.req_dataptr();

	np = buff.get_bis().get_block_dims(idxp).get_dim(1);

	double tmp2 = *(pp + posp0 * np + posp1);


        ct2.ret_dataptr(pp);
        ctrl.ret_block(idxp);

	tmp -=  tmp1 * tmp2;
	
	// now tmp has V_{ij} - L{ij} * L_{ik}

	}
	}// end of sumation over k
	
	// divide by diagonal
	
	idxp[0] = 0;
        idxp[1] = idx[1];
	posp0 = 0;
	posp1 = pos1;

        for(size_t m = 0; m < i1; m++)
        {
	posp0++;
        if(buff.get_bis().get_block_dims(idxp).get_dim(0) - 1 < posp0)
        {
                posp0 -= buff.get_bis().get_block_dims(idxp).get_dim(0);
                idxp[0]++;
        }
        // should it be else here?
        }

        tensor_i<2 ,double> &t3 = ctrl.req_block(idxp); 
	tensor_ctrl<2, double> ct3(t3);
        pp = ct3.req_dataptr();

	np = buff.get_bis().get_block_dims(idxp).get_dim(1);

	tmp = tmp / (*(pp + posp0 * np + posp1));


	// now tmp has V_{ij} - \sum_{k=1}^{i-1} L_{jk} L_{ik}/L_{ii}

	ct3.ret_dataptr(pp);
        ctrl.ret_block(idxp);

        tensor_i<2 ,double> &t4 = ctrl.req_block(idx); 
	tensor_ctrl<2, double> ct4(t4);
        p = ct4.req_dataptr();
	
	np = buff.get_bis().get_block_dims(idx).get_dim(1);

	*(p + pos0 * np + pos1) = tmp;

        ct4.ret_dataptr(p);
        ctrl.ret_block(idx);

	// end of the step 1.

	//last step - update diagonals
	// V_{jj} = V_{jj} - L^2{ji}
	
	idxp[0]=idx[0];
	posp0 = pos0;
	idxp[1] = 0;
	posp1 = 0;

	for(size_t m = 0;m < i0 ; m++)
	{
	posp1++;
	if(buff.get_bis().get_block_dims(idxp).get_dim(1) - 1 < posp1)
        {
                posp1 -= buff.get_bis().get_block_dims(idxp).get_dim(1);
                idxp[1]++;
        }
        // should it be else here?
	}	

	np =  buff.get_bis().get_block_dims(idxp).get_dim(1);

        tensor_i<2 ,double> &t5 = ctrl.req_block(idxp); 
	tensor_ctrl<2, double> ct5(t5);
        pp = ct5.req_dataptr();

	tmp = *(pp + posp0 * np + posp1);


	// now tmp has diagonal V_{jj}

        ct5.ret_dataptr(pp);
        ctrl.ret_block(idxp);

        tensor_i<2 ,double> &t6 = ctrl.req_block(idx); 
	tensor_ctrl<2, double> ct6(t6);
        p = ct6.req_dataptr();

	np =  buff.get_bis().get_block_dims(idx).get_dim(1);

	double tmp2 = *(p + pos0 * np + pos1);

	tmp -= tmp2 * tmp2;

	// now tmp has V_{jj} - L_{ji}^2

        ct6.ret_dataptr(p);
        ctrl.ret_block(idx);


        np =  buff.get_bis().get_block_dims(idxp).get_dim(1);
        tensor_i<2 ,double> &t7 = ctrl.req_block(idxp); 
	tensor_ctrl<2, double> ct7(t7);
        pp = ct7.req_dataptr();

	*(pp + posp0 * np + posp1) = tmp;

        ct7.ret_dataptr(pp);
        ctrl.ret_block(idxp);
	
	}

	pos0++;
	}

	m_rank++;
	pos1++;
	
	std::stringstream os;
        os<<std::endl;
	#ifdef PRINT
	std::cout<<"The matrix for the rank = "<<m_rank<<std::endl;
	btod_print<2>(os).perform(buff);
	std::cout<<os.str()<<std::endl;
	#endif

	}


	// apply permutation back
	
}

void cholesky::perform(block_tensor_i<2, double> &btb)
{
	
	size_t n = btb.get_bis().get_dims().get_dim(0);
	size_t R = m_rank ;
	
	block_tensor_i<2, double> &buff(*pta);

	// Should be removed!
	//btod_copy<2>(buff).perform(btb);
	//btod_set<2>(3).perform(btb);
	
	index<2> idxi;
	idxi[0] = 0;

        index<2> idxo;
        idxo[0] = 0;

	size_t posi0 = 0;
	size_t poso0 = 0;
	
	for(size_t i0 = 0; i0 < n;i0++)
	{

	if(buff.get_bis().get_block_dims(idxi).get_dim(0) - 1 < posi0)
	{
		posi0 -= buff.get_bis().get_block_dims(idxi).get_dim(0);
		idxi[0]++;
	}

        if(btb.get_bis().get_block_dims(idxo).get_dim(0) - 1 < poso0)
        {
                poso0 -= btb.get_bis().get_block_dims(idxo).get_dim(0);
                idxo[0]++;
        }
	
	idxi[1] = 0;
	idxo[1] = 0;
	size_t posi1 = 0;
	size_t poso1 = 0;

	for(size_t i1 = 0; i1 < R; i1++)
	{
		if(buff.get_bis().get_block_dims(idxi).get_dim(1) - 1 < posi1)
		{
			posi1 -= buff.get_bis().get_block_dims(idxi).get_dim(1);
			idxi[1]++;
		}
		
		if(btb.get_bis().get_block_dims(idxo).get_dim(1) - 1 < poso1)
                {
                        poso1 -= btb.get_bis().get_block_dims(idxo).get_dim(1);
                        idxo[1]++;
                }

		block_tensor_ctrl<2, double> ctrli(buff);
		block_tensor_ctrl<2, double> ctrlo(btb);

		if(ctrli.req_is_zero_block(idxi)==false)
		{
	
		
		tensor_i<2, double> &ti = ctrli.req_block(idxi);
		tensor_ctrl<2, double> cti(ti);
		const double *pi = cti.req_const_dataptr();		
		
		tensor_i<2, double> &to = ctrlo.req_block(idxo);
		tensor_ctrl<2, double> cto(to);
		double *po = cto.req_dataptr();

		size_t Rp = btb.get_bis().get_block_dims(idxo).get_dim(1); 
		size_t np = buff.get_bis().get_block_dims(idxi).get_dim(1);

		*(po +poso0 * Rp + poso1) = *(pi + posi0 * np + posi1);

		cti.ret_const_dataptr(pi);
		cto.ret_dataptr(po);
		pi=0;
		po=0;

		ctrli.ret_block(idxi);
		ctrlo.ret_block(idxo);
		
		}
		posi1++;
		poso1++;
	}
	
	posi0++;
	poso0++;
	}

}


}//namespace libtensor

