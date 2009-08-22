#ifndef LIBTENSOR_BTOD_RANDOM_H
#define LIBTENSOR_BTOD_RANDOM_H

#include <list>
#include <map>
#include <utility>
#include <libvmm.h>
#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"

namespace libtensor {

/**	\brief Fills a block %tensor with random data without affecting its
		%symmetry

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_random {
public:
	void perform(block_tensor_i<N, double> &bt) throw(exception);
	void perform(block_tensor_i<N, double> &bt, const index<N> &blk ) 
		throw(exception);
private:

	typedef std::pair< index<N>,transf<N,double> > pair_t;
	typedef std::list< pair_t > index_list_t;
	typedef std::list< transf<N,double> > list_t;
	/** \brief Determines all transformation which can be applied on a block of 
  			   the given orbit 
  
 		\param reslist resulting list of transformations 
 		\param index_list list of actual indexes and transformations 
 		\param sym symmetry object to obtain symmetry elements
 		
 	**/
	void determine_transformations( list_t& reslist, index_list_t& index_list, 
		const symmetry<N,double>& sym ); 
};

template<size_t N>
void btod_random<N>::determine_transformations( list_t& reslist, 
			index_list_t& index_list, const symmetry<N,double>& sym ) 
{
	// for each symmetry element
	size_t nsymels=sym.get_num_elements();  
	for ( size_t i=0; i<nsymels; i++ ) {
		// create a new pair
		index<N> new_idx(index_list.rbegin()->first);
		transf<N,double> new_tr;
		sym.get_element(i).apply(new_idx,new_tr);
			
		typename index_list_t::iterator it=index_list.begin();
		while ( it != index_list.end() ) {
			if ( new_idx.equals(it->first) ) break;
			it++;
		} 
		if ( it != index_list.end() ) {
			transf<N,double> tr;
			it++;
			while (it != index_list.end() ) { 
				tr.transform(it->second);
				it++;				
			}
			tr.transform(new_tr);
			
			if ( tr != transf<N,double>() ) {
				typename list_t::iterator list_it=reslist.begin();
				while ( list_it != reslist.end() ) {
					if ( tr == *list_it ) break;
					list_it++;
				}
				if ( list_it == reslist.end() ) 
					reslist.push_back(tr);
			}
		}
		else {
			index_list.push_back(pair_t(new_idx,new_tr));
			determine_transformations(reslist,index_list,sym);
			index_list.pop_back();
		}
	}		
}

template<size_t N>
void btod_random<N>::perform(block_tensor_i<N, double> &bt) 
	throw(exception) 
{
	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	block_tensor_ctrl<N, double> ctrl(bt);

	tod_random<N> randr;
	
	orbit_list<N,double> orblist(ctrl.req_symmetry());
	for(typename orbit_list<N,double>::iterator it=orblist.begin(); it!=orblist.end(); it++) {
		orbit<N, double> orb(ctrl.req_symmetry(),*it);
		index<N> blkidx;
		bidims.abs_index(orb.get_abs_canonical_index(), blkidx);
		
		list_t transf_list;
		index_list_t start_list;
		start_list.push_back( pair_t(blkidx,transf<N,double>()) );
		determine_transformations(transf_list,start_list,ctrl.req_symmetry());
		
		tensor_i<N, double> &blk = ctrl.req_block(blkidx);
		
		if ( transf_list.empty() ) {
			randr.perform(blk);
		}
		else {
			tensor<N,double,libvmm::std_allocator<double> > tmp(blk.get_dims());
			tensor_i<N,double> *ta, *tb, *tc;
			ta=&tmp;
			randr.perform(*ta);
			tb=&blk;
				
			typename std::list<transf<N,double> >::iterator it=transf_list.begin();
			for ( ; it!=transf_list.end(); it++ ) {
				tod_add<N> doadd(*ta,0.5);
				doadd.add_op(*ta,it->get_perm(),0.5*it->get_coeff());
				doadd.perform(*tb);
				tc=ta;
				ta=tb;
				tb=tc;
			}
			
			if ( ta != &blk ) {
				tod_copy<N> finalcopy(*ta,1.0);
				finalcopy.perform(blk);
			}
		}
	}
}

template<size_t N>
void btod_random<N>::perform(block_tensor_i<N, double> &bt, 
	const index<N> &blkidx ) throw(exception) 
{
	block_tensor_ctrl<N, double> ctrl(bt);
	tod_random<N> randr;
	
	orbit<N, double> orb(ctrl.req_symmetry(),blkidx);
		
	list_t transf_list;
	index_list_t start_list;
	start_list.push_back( pair_t(blkidx,transf<N,double>()) );
	determine_transformations(transf_list,start_list,ctrl.req_symmetry());
		
	tensor_i<N, double> &blk = ctrl.req_block(blkidx);
		
	if ( transf_list.empty() ) {
		randr.perform(blk);
	}
	else {
		tensor<N,double,libvmm::std_allocator<double> > tmp(blk.get_dims());
		tensor_i<N,double> *ta, *tb, *tc;
		ta=&tmp;
		randr.perform(*ta);
		tb=&blk;
				
		typename std::list<transf<N,double> >::iterator it=transf_list.begin();
		for ( ; it!=transf_list.end(); it++ ) {
			tod_add<N> doadd(*ta,0.5);
			doadd.add_op(*ta,it->get_perm(),0.5*it->get_coeff());
			doadd.perform(*tb);
			tc=ta;
			ta=tb;
			tb=tc;
		}
			
		if ( ta != &blk ) {
			tod_copy<N> finalcopy(*ta,1.0);
			finalcopy.perform(blk);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_H
