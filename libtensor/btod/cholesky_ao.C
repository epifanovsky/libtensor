#include <sstream>
#include "../core/allocator.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/block_tensor.h"
#include "scalar_transf_double.h"
#include "btod_add.h"
#include "btod_contract2.h"
#include "btod_copy.h"
#include "btod_diag.h"
#include "cholesky_ao.h"
#include "btod_extract.h"
#include "btod_scale.h"
#include "btod_set_diag.h"
#include "btod_set_elem.h"
#include "btod_set.h"
#include "btod_select.h"
#include "btod_mult.h"

#include "btod_print.h"

//#define PRINT

namespace libtensor
{

cholesky_ao::cholesky_ao(block_tensor_i<4, double> &bta, double tol) :
m_bta(bta) ,m_tol(tol), m_rank(0)
, chol_vecs (new list), diagonal(NULL), column (NULL)
{

}

cholesky_ao::~cholesky_ao(){     
       if(chol_vecs!=NULL) delete_chol_vecs();
       diagonal = NULL;
       column = NULL;
}

void cholesky_ao::delete_chol_vecs()
{
	if(chol_vecs==NULL) return;
	for(size_t i = 0; i < chol_vecs->size(); i++)
	{
	        block_tensor<2, double, std_allocator<double> > * vector = (*chol_vecs)[i];
                delete vector;
		vector = NULL;
	}
	delete chol_vecs;
	chol_vecs = NULL;
}

void cholesky_ao::decompose()
{
	get_diag();

	if(diagonal == NULL)
	{
	std::cout<<"Error in extracting diagonal! Program terminated"<<std::endl;
	exit(2);
	}

	index<2> idxdibl; // index inside block of diagonal
        idxdibl[0] = 0; idxdibl[1] = 0;

        index<2> idxdbl; // index of the block of diagonal
        idxdbl[0] = 0; idxdbl[1] = 0;

        double max_diag = 0;
        max_diag = sort_diag(idxdbl, idxdibl);

	//std::cout<<"Maximum diagonal element is "<<max_diag<<std::endl;

	while(max_diag > m_tol)
	{

	// extract column
	
	index<4> idxibl; // index inside block
        idxibl[0] = 0; idxibl[1] = 0; idxibl[2] = idxdibl[0]; idxibl[3] = idxdibl[1];

        index<4> idxbl; // index of the block
        idxbl[0] = 0; idxbl[1] = 0; idxbl[2] = idxdbl[0]; idxbl[3] = idxdbl[1];
	
	extract_column(idxbl, idxibl);
	
	// compute residual
	
	size_t chol_length = (*chol_vecs).size();

	if(chol_length!=m_rank)
	{
	std::cout<<"Error! The number of cholesky vectors does not correspond to rank. Program terminated."<<std::endl;
	exit(2);
	}

	for(size_t i = 0; i < chol_length; i++)
	{
		//run over all already computed cholesky vectors and subtract it from computed column
		block_tensor_i<2, double> & chol_vector (*((*chol_vecs)[i]));

		// compute coefficient
		double c = 0;
		
		block_tensor_ctrl<2, double> ctrl(chol_vector);
		dense_tensor_i<2, double> &t = ctrl.req_block(idxdbl);
		dense_tensor_ctrl<2, double> ct(t);
        	double *p = ct.req_dataptr();	

		size_t np = chol_vector.get_bis().get_block_dims(idxdbl).get_dim(1);

		c = *(p + idxdibl[0] * np + idxdibl[1]);
		
	        ct.ret_dataptr(p);
	        ctrl.ret_block(idxdbl);
	
		// subtract previously obtained cholesky vectors from the column

		btod_copy<2>(chol_vector).perform(*column , -c);
	}
	// compute and add cholesky vector to the list
	
	btod_scale<2>(*column, 1/sqrt(max_diag)).perform();
	chol_vecs->push_back(column);

	// update diagonal
	
	size_t last_elem = chol_vecs->size() - 1;
	
	block_tensor_i<2, double> & chol_vector (*((*chol_vecs)[last_elem]));
	
	btod_mult<2> mult (chol_vector, chol_vector);

	block_tensor<2, double, std_allocator <double> > * tmp = 
		new block_tensor<2, double, std_allocator <double> > (mult.get_bis());
	mult.perform(*tmp);
	btod_copy<2>(*tmp).perform(*diagonal , -1);

	delete tmp;
	tmp = NULL;

	//calculate new maximim diagonal element
	
	max_diag = sort_diag(idxdbl, idxdibl);

	m_rank++;

	column = NULL;
	}

	delete diagonal;
	diagonal = NULL;
}

void cholesky_ao::extract_column(index<4> &idxbl, index<4> &idxibl)
{
	// extract column from the btensor. Will be changed to direct computation
	mask<4> msk;
        msk[0] = true; msk[1] = true; msk[2] = false; msk[3] = false;

        btod_extract<4, 2> ex(m_bta , msk, idxbl, idxibl);
	column = new block_tensor<2, double, std_allocator <double> > (ex.get_bis());
	ex.perform(*column);
}

void cholesky_ao::get_diag()
{
	// extract diagonal from the btensor. Will be changed to direct computation
	mask<4> msk1;
	msk1[0] = true; msk1[2] = true;
	btod_diag <4, 2> diag1 (m_bta, msk1);
	block_tensor<3, double, std_allocator <double> > * diaginterm = new block_tensor<3, double, std_allocator <double> > (diag1.get_bis());
	diag1.perform(*diaginterm);
	
	mask<3> msk2;
	msk2[1] = true; msk2[2] = true;
	btod_diag <3, 2> diag2 (*diaginterm , msk2);
	diagonal = new block_tensor<2, double, std_allocator <double> > (diag2.get_bis());
	diag2.perform(*diagonal);

	delete diaginterm;
	diaginterm = NULL;
}

double cholesky_ao::sort_diag(index<2> &idxbl, index<2> &idxibl)
{
	#ifdef PRINT
        std::stringstream os;
        os<<std::endl;	
	os<<"Diagonal is "<<std::endl;
	btod_print<1>(os).perform(*diagonal);
        #endif

	double maxdiag = 0;
	btod_select<2, compare4absmax>::list_t lst1;
	btod_select<2, compare4absmax>(*diagonal).perform(lst1, 1);

	if(lst1.begin() != lst1.end()) {
		maxdiag = lst1.begin()->value;
                index<2> bidx;
                bidx = lst1.begin()->bidx;
                index<2> idx;
                idx = lst1.begin()->idx;
                idxbl[0] = bidx[0]; idxbl[1] = bidx[1];
               	idxibl[0] = idx[0]; idxibl[1] = idx[1];
                // here it might be a bug if
                // bis of diag not equal to the
                // bis of the columns of the
                // matrix
                }

	return maxdiag;
}

void cholesky_ao::perform(block_tensor_i<3, double> &btb)
{		
	index<3> idxibl; // index inside block of diagonal
        idxibl[0] = 0; idxibl[1] = 0; idxibl[2] = 0;

        index<3> idxbl; // index of the block of diagonal
        idxbl[0] = 0; idxbl[1] = 0; idxbl[2] = 0;

	size_t n0 = btb.get_bis().get_dims().get_dim(0);
	size_t n1 = btb.get_bis().get_dims().get_dim(1);
	size_t n2 = btb.get_bis().get_dims().get_dim(2);

	if(n2 != m_rank)
	{
	std::cout<<"Error! Number of cholesky vectors and rank are different! Terminated. "<<std::endl;
	exit(2);
	}

	for(size_t i2 = 0; i2 < n2; i2++)
	{//index i2 start
		block_tensor_i<2, double> & chol_vector (*((*chol_vecs)[i2]));
		if(btb.get_bis().get_block_dims(idxbl).get_dim(2) - 1 < idxibl[2] )
		{
		idxibl[2] -= btb.get_bis().get_block_dims(idxbl).get_dim(2);
		idxbl[2]++;
		}
		idxibl[1] = 0;
                idxbl[1] = 0;
		for(size_t i1 = 0; i1 < n1; i1++)
		{//index i1 start
			if(btb.get_bis().get_block_dims(idxbl).get_dim(1) - 1 < idxibl[1] )
		      	{
                	idxibl[1] -= btb.get_bis().get_block_dims(idxbl).get_dim(1);
                	idxbl[1]++;
                	}
			idxibl[0] = 0;
			idxbl[0] = 0;
			for(size_t i0 = 0;i0 < n0; i0++)
			{//index i0 start
				if(btb.get_bis().get_block_dims(idxbl).get_dim(0) - 1 < idxibl[0] )
                        	{
                        	idxibl[0] -= btb.get_bis().get_block_dims(idxbl).get_dim(0);
                        	idxbl[0]++;
                        	}
				block_tensor_ctrl<2, double> ctrl_chol(chol_vector);
                		block_tensor_ctrl<3, double> ctrl(btb);	

				index<2> idxchol, idxichol;
				idxchol[0] = idxbl[0]; idxchol[1] = idxbl[1];
				idxichol[0] = idxibl[0]; idxichol[1] = idxibl[1];

				dense_tensor_i<2, double> &tchol = ctrl_chol.req_block(idxchol);
                		dense_tensor_ctrl<2, double> ctchol(tchol);
                		const double *pchol = ctchol.req_const_dataptr();
				
				dense_tensor_i<3, double> &t = ctrl.req_block(idxbl);
                                dense_tensor_ctrl<3, double> ct(t);
                                double *p = ct.req_dataptr();

				size_t nbtb1 = btb.get_bis().get_block_dims(idxbl).get_dim(1);
				size_t nbtb2 = btb.get_bis().get_block_dims(idxbl).get_dim(2);
				size_t nchol1 = chol_vector.get_bis().get_block_dims(idxchol).get_dim(1);

				*(p + idxibl[0] * nbtb1 * nbtb2 + idxibl[1] * nbtb2 + idxibl[2]) = 
					*(pchol + idxichol[0] * nchol1 + idxichol[1]);

                		ctchol.ret_const_dataptr(pchol);
                		ct.ret_dataptr(p);
                		pchol=0;
                		p=0;

                		ctrl_chol.ret_block(idxchol);
                		ctrl.ret_block(idxbl);
		
				idxibl[0]++;
			}//index i0 end
			idxibl[1]++;
		}//index i1 end
		idxibl[2]++;
	}//index i2 end
	
}

}//namespace libtensor

