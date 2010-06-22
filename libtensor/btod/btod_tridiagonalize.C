#include <iostream>
#include <libtensor/libtensor.h>
#include <libvmm/libvmm.h>
#include "../core/block_tensor_ctrl.h"
#include "btod_extract.h"
#include "btod_tridiagonalize.h"


namespace libtensor
{
btod_tridiagonalize::btod_tridiagonalize(block_tensor_i<2, double> &bta) :
	m_bta(bta){

}

void btod_tridiagonalize::perform(block_tensor_i<2, double> &btb,
		block_tensor_i<2, double> &S)
	{
	//copy the input tensor to the out tensor
	btod_set<2>(0).perform(btb);
	btod_copy<2>(m_bta).perform(btb);

	const dimensions<2> *dimsa;
	dimsa = &(m_bta.get_bis().get_dims());

	//creates a column block tensor
	index<1> i1a, i1b;
	i1b[0] = (*dimsa).get_dim(0) - 1;
	dimensions<1> dims(index_range<1>(i1a, i1b));
	block_index_space<1> bis(dims);

	//splitting
	mask<1> splmskc;
	splmskc[0]=true;
	const split_points *splpts;
	splpts = &(m_bta.get_bis().get_splits(m_bta.get_bis().get_type(0)));
	for(size_t j=0; j < (*splpts).get_num_points();j++)
	{
		size_t f;
		f = (*splpts)[j];
		bis.split(splmskc,f);
	}

	typedef libvmm::std_allocator<double> allocator_t;
	block_tensor<1, double, allocator_t> btcol(bis);//tensor which contains
	//column

	block_tensor_ctrl<1, double> cab(btcol);
	block_tensor<1, double, allocator_t> v(bis);
	block_tensor_ctrl<1, double> cabv(v);

	block_tensor<2, double, allocator_t> P(btb.get_bis());
	block_tensor<2, double, allocator_t> vv(btb.get_bis());
	block_tensor<2, double, allocator_t> temp(btb.get_bis());

	btod_set<2>(0).perform(S);
	btod_set_diag<2> (1).perform(S);

	size_t pos;
	size_t size = (*dimsa).get_dim(1);

	index<2> idxbl;
	idxbl[0] = 0; idxbl[1] = 0;
	index<2> idxibl;
	size_t counterc=0;

	//start the series of Householder's refelections
	for (size_t counter =0;counter < size - 2  ;counter++)
	{
		btod_set<1>(0).perform(v);
		btod_set<1>(0).perform(btcol);

		btod_set<2>(0).perform(P);
		btod_set_diag<2> (1).perform(P);
		btod_set<2>(0).perform(vv);
		btod_set<2>(0).perform(temp);

		mask<2> msk;
		msk[0] = true; msk[1] = false;

		dimensions<2> dimsiblock(m_bta.get_bis().get_block_dims(idxbl));
		size_t iblock = dimsiblock.get_dim(1);

		if(counterc > iblock - 1)
		{
			counterc = 0;
			idxbl[1]++;
		}

		idxibl[0] = 0; idxibl[1] = counterc;

		//put the column of bta into the btcol
		btod_extract<2, 1>(btb, msk, idxbl, idxibl).perform(btcol);

		//requesting the pointer for reading
		index<1> idx;
		idx[0] = 0;

		//compute alpha and sum (norm)
		double alpha;
		double sum=0;
		pos = counter + 1;
		bool done = 0;
		while(!done)
		{
			if(btcol.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
			{
				pos -= btcol.get_bis().get_block_dims(idx).get_dim(0);
				idx[0]++;
			}
			else
			{
				done = 1;
			}
		}
		for(size_t i = counter + 1;i < size;i++)
		{
			//position of the pointer inside the block
				if(btcol.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
			{
				pos = 0;
				idx[0]++;
			}

			if(cab.req_is_zero_block(idx)==false)
			{
				tensor_i<1 ,double> &tcol = cab.req_block(idx);
				tensor_ctrl<1, double> ca(tcol);

				const double *pa = ca.req_const_dataptr();
			sum += (*(pa + pos)) * (*(pa + pos));
			ca.ret_dataptr(pa);
			pa=0;
			cab.ret_block(idx);
			}

			pos++;
		}

		done = 0;
		idx[0] = 0;
		pos = counter + 1;
		double a;
		while(!done)
		{
			if(btcol.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
			{
				pos -= btcol.get_bis().get_block_dims(idx).get_dim(0);
				idx[0]++;
			}
			else
			{
				done = 1;
			}
		}

		tensor_i<1 ,double> &tcol = cab.req_block(idx);
		tensor_ctrl<1, double> ca(tcol);

		const double *pa = ca.req_const_dataptr();

		a = *(pa + pos);

		ca.ret_dataptr(pa);
		pa=0;
		cab.ret_block(idx);

		if(a>=0.0)
		{
			alpha = - sqrt(sum);
		}
		else
		{
			alpha = sqrt(sum);
		}

		//compute r
		double r;
		//r = (alpha*alpha - a * alpha );
		r=sqrt((alpha*alpha - a * alpha )/2);

		//fill v (vector of Householder's reflections)  with a data
		done = 0;
		idx[0] = 0;
		pos = counter + 1;

		while(!done)
		{
			if(btcol.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
			{
				pos -= btcol.get_bis().get_block_dims(idx).get_dim(0);
				idx[0]++;
			}
			else
			{
				done = 1;
			}
		}

		index<1> idxibl1;

		for(size_t i = 0;i < pos;i++)
		{
			idxibl1[0]=i;
			btod_set_elem<1>().perform(v,idx,idxibl1,0);
		}

		idxibl1[0] = pos;

		btod_set_elem<1>().perform(v,idx,idxibl1,a - alpha);

			pos++;

		for(size_t i = counter + 2; i < size;i++)
		{
			if(btcol.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
			{
				idx[0]++;
				pos = 0;
			}


			if(cab.req_is_zero_block(idx)==1)
			{
				cabv.req_zero_block(idx);
			}
			else
			{
				tensor_i<1 ,double> &tcol1 = cab.req_block(idx);
				tensor_ctrl<1, double> ca1(tcol1);
				const double *pa1 = ca1.req_const_dataptr();
				idxibl1[0]=pos;
				btod_set_elem<1>().perform(v,idx,idxibl1,*(pa1 + pos));
				ca1.ret_dataptr(pa1);
				pa1=0;
				cab.ret_block(idx);
			}

			pos++;
		}

		counterc++;
		btod_scale<1> (v,1/(2*r)).perform();

		//Create v*vt
		contraction2<1,1,0> contrvv;
		btod_contract2<1,1,0>(contrvv,v,v).perform(vv);

		//Create matrix of Householder's reflections P
		btod_add<2> opv(P);
		opv.add_op(vv,-2);
		opv.perform(temp);
		btod_copy<2>(temp).perform(P);

		//Apply matrix of Householder's reflections on the tensor
		contraction2<1,1,1> contrp;
		contrp.contract(1,0);
		btod_contract2<1,1,1>(contrp,P,btb).perform(temp);
		btod_contract2<1,1,1>(contrp,temp,P).perform(btb);

		//Update matrix of transformation
		btod_contract2<1,1,1>(contrp,S,P).perform(temp);
		btod_copy<2>(temp).perform(S);
	}

}

void btod_tridiagonalize::print(block_tensor_i<2, double> &btb)
{
	//operation prints the initial matrix m_bta and tridiagonal matrix btb
	size_t size = m_bta.get_bis().get_dims().get_dim(1);
	std::cout<<"The matrix A is:"<<std::endl;
	std::cout<<std::endl;
	index<2> idxi;
	idxi[0] = 0;
	int posv = 0;

	for(size_t i =0;i < size;i++)
	{
		if(m_bta.get_bis().get_block_dims(idxi).get_dim(0) - 1 < posv)
		{
			posv -= m_bta.get_bis().get_block_dims(idxi).get_dim(0);
			idxi[0]++;
		}

		idxi[1] = 0;
		int posh = 0;

		for(size_t j =0;j < size;j++)
		{
			if(m_bta.get_bis().get_block_dims(idxi).get_dim(1) - 1 < posh)
			{
				posh -= m_bta.get_bis().get_block_dims(idxi).get_dim(1);
				idxi[1]++;
			}
			block_tensor_ctrl<2, double> ctrl(m_bta);
			if(ctrl.req_is_zero_block(idxi)==false)
			{
			tensor_i<2 ,double> &tbtb = ctrl.req_block(idxi);
			tensor_ctrl<2, double> catrl(tbtb);
			const double *pa = catrl.req_const_dataptr();
			std::cout<<*(pa + posv * m_bta.get_bis().get_block_dims(idxi)
					.get_dim(1) + posh)<<" ";
			catrl.ret_dataptr(pa);
			pa=0;
			ctrl.ret_block(idxi);
			}
			else
			{
				std::cout<<"'"<<" ";
			}
			posh++;
		}

		std::cout<<std::endl;
		posv++;
	}

	std::cout<<std::endl;

	idxi[0] = 0;
	posv = 0;

	std::cout<<"The tridiagonal matrix for matrix A is:"<<std::endl;
	std::cout<<std::endl;

	for(size_t i =0;i < size;i++)
	{
		if(btb.get_bis().get_block_dims(idxi).get_dim(0) - 1 < posv)
		{
			posv -= btb.get_bis().get_block_dims(idxi).get_dim(0);
			idxi[0]++;
		}

		idxi[1] = 0;
		int posh = 0;

		for(size_t j =0;j < size;j++)
		{
			if(btb.get_bis().get_block_dims(idxi).get_dim(1) - 1 < posh)
			{
				posh -= btb.get_bis().get_block_dims(idxi).get_dim(1);
				idxi[1]++;
			}
			block_tensor_ctrl<2, double> ctrl(btb);
			if(ctrl.req_is_zero_block(idxi)==false)
			{
			tensor_i<2 ,double> &tbtb = ctrl.req_block(idxi);
			tensor_ctrl<2, double> catrl(tbtb);
			const double *pa = catrl.req_const_dataptr();
			std::cout<<*(pa + posv * btb.get_bis().get_block_dims(idxi).
					get_dim(1) + posh)<<" ";
			catrl.ret_dataptr(pa);
			pa=0;
			ctrl.ret_block(idxi);
			}
			else
			{
				std::cout<<"'"<<" ";
			}
			posh++;
		}

		std::cout<<std::endl;
		posv++;
	}

	std::cout<<std::endl;
	std::cout<<"================================================"<<std::endl;
	std::cout<<std::endl;

}

}//namespace libtensor

