#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include <list>
#include <map>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/sequence.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/so_proj_down.h"
#include "../symmetry/so_proj_up.h"
#include "../symmetry/so_union.h"
#include "../tod/contraction2.h"
#include "../tod/tod_contract2.h"
#include "../tod/tod_sum.h"
#include "additive_btod.h"
#include "../not_implemented.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Contraction of two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 :
	public additive_btod<N + M>,
	public timings< btod_contract2<N, M, K> > {

public:
	static const char *k_clazz; //!< Class name

private:
	static const size_t k_ordera = N + K; //!< Order of first argument (a)
	static const size_t k_orderb = M + K; //!< Order of second argument (b)
	static const size_t k_orderc = N + M; //!< Order of result (c)
	static const size_t k_totidx = N + M + K; //!< Total number of indexes
	static const size_t k_maxconn = 2 * k_totidx; //!< Index connections

private:
	typedef struct block_contr {
	public:
		size_t m_absidxa;
		size_t m_absidxb;
		double m_c;
		permutation<k_ordera> m_perma;
		permutation<k_orderb> m_permb;

	public:
		block_contr(size_t aia, size_t aib, double c,
			const permutation<k_ordera> &perma,
			const permutation<k_orderb> &permb)
		: m_absidxa(aia), m_absidxb(aib), m_c(c), m_perma(perma),
			m_permb(permb)
		{ }
		bool is_same_perm(const transf<k_ordera, double> &tra,
			const transf<k_orderb, double> &trb) {

			return m_perma.equals(tra.get_perm()) &&
				m_permb.equals(trb.get_perm());
		}
	} block_contr_t;
	typedef std::list<block_contr_t> block_contr_list_t;
	typedef std::map<size_t, block_contr_list_t*> schedule_t;

private:
	contraction2<N, M, K> m_contr; //!< Contraction
	block_tensor_i<k_ordera, double> &m_bta; //!< First argument (A)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (B)
	block_index_space<k_orderc> m_bis; //!< Block %index space of the result
	symmetry<k_orderc, double> m_sym; //!< Symmetry of the result
	dimensions<k_ordera> m_bidimsa; //!< Block %index dims of A
	dimensions<k_orderb> m_bidimsb; //!< Block %index dims of B
	dimensions<k_orderc> m_bidimsc; //!< Block %index dims of the result
	schedule_t m_contr_sch; //!< Contraction schedule
	assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation
		\param contr Contraction.
		\param bta Block %tensor A (first argument).
		\param btb Block %tensor B (second argument).
	 **/
	btod_contract2(const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_contract2();

	//@}

	//!	\name Implementation of
	//		libtensor::direct_block_tensor_operation<N + M, double>
	//@{

	virtual const block_index_space<N + M> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N + M, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N + M, double> &get_schedule() const {
		return m_sch;
	}

	//@}

	using additive_btod<N + M>::perform;

protected:
	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i);
	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i, const transf<N + M, double> &tr,
		double c);

private:
	static block_index_space<N + M> make_bis(
		const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);
	void make_symmetry();
	void make_schedule();

	/**	\brief For an orbit in a and b, make a list of blocks in c
	 **/
	void make_schedule(const orbit<k_ordera, double> &oa,
		const orbit<k_orderb, double> &ob,
		const orbit_list<k_orderc, double> &olc);

	void clear_schedule(schedule_t &sch);

	void contract_block(
		block_contr_list_t &lst, const index<k_orderc> &idxc,
		block_tensor_ctrl<k_ordera, double> &ctrla,
		block_tensor_ctrl<k_orderb, double> &ctrlb,
		tensor_i<k_orderc, double> &blkc,
		const transf<k_orderc, double> &trc,
		bool zero, double c);

private:
	btod_contract2(const btod_contract2<N, M, K>&);
	btod_contract2<N, M, K> &operator=(const btod_contract2<N, M, K>&);

};


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb) :

	m_contr(contr), m_bta(bta), m_btb(btb),
	m_bis(make_bis(contr, bta, btb)), m_sym(m_bis),
	m_bidimsa(m_bta.get_bis().get_block_index_dims()),
	m_bidimsb(m_btb.get_bis().get_block_index_dims()),
	m_bidimsc(m_bis.get_block_index_dims()), m_sch(m_bidimsc) {

	make_symmetry();
	make_schedule();
}


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::~btod_contract2() {

	clear_schedule(m_contr_sch);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(tensor_i<N + M, double> &blk,
	const index<N + M> &i) {

	static const char *method =
		"compute_block(tensor_i<N + M, double>&, const index<N + M>&)";

	btod_contract2<N, M, K>::start_timer();

	try {

		block_tensor_ctrl<k_ordera, double> ca(m_bta);
		block_tensor_ctrl<k_orderb, double> cb(m_btb);

		abs_index<k_orderc> aic(i, m_bidimsc);
		typename schedule_t::iterator isch =
			m_contr_sch.find(aic.get_abs_index());
		if(isch == m_contr_sch.end()) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "i");
		}

		transf<k_orderc, double> trc0;
		contract_block(*isch->second, aic.get_index(), ca, cb,
			blk, trc0, true, 1.0);

	} catch(...) {
		btod_contract2<N, M, K>::stop_timer();
		throw;
	}

	btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(tensor_i<N + M, double> &blk,
	const index<N + M> &i, const transf<N + M, double> &tr, double c) {

	static const char *method = "compute_block(tensor_i<N + M, double>&, "
		"const index<N + M>&, const transf<N + M, double>&, double)";

	btod_contract2<N, M, K>::start_timer();

	try {

		block_tensor_ctrl<k_ordera, double> ca(m_bta);
		block_tensor_ctrl<k_orderb, double> cb(m_btb);

		abs_index<k_orderc> aic(i, m_bidimsc);
		typename schedule_t::iterator isch =
			m_contr_sch.find(aic.get_abs_index());
		if(isch == m_contr_sch.end()) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "i");
		}

		contract_block(*isch->second, aic.get_index(), ca, cb,
			blk, tr, false, c);
	} catch(...) {
		btod_contract2<N, M, K>::stop_timer();
		throw;
	}

	btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
block_index_space<N + M> btod_contract2<N, M, K>::make_bis(
	const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb) {

	static const char *method = "make_bis()";

	const block_index_space<k_ordera> &bisa = bta.get_bis();
	const block_index_space<k_orderb> &bisb = btb.get_bis();

	//	Check if contracted indexes are compatible

	const sequence<k_maxconn, size_t> &conn = contr.get_conn();
	for(size_t idima = 0; idima < k_ordera; idima++) {
		size_t iconn = conn[k_orderc + idima];
		if(iconn >= k_orderc + k_ordera) {
			size_t idimb = conn[k_orderc + idima] -
				k_orderc - k_ordera;
			size_t itypa = bisa.get_type(idima);
			size_t itypb = bisb.get_type(idimb);
			if(!bisa.get_splits(itypa).equals(
				bisb.get_splits(itypb))) {
				throw bad_parameter(g_ns, k_clazz, method,
					__FILE__, __LINE__,
					"Block tensor dimensions are unsuitable"
					" for contraction.");
			}
		}
	}

	//	Build the result block index space

	index<k_orderc> i0, i1;
	for(size_t idimc = 0; idimc < k_orderc; idimc++) {
		size_t iconn = conn[idimc];
		if(iconn >= k_orderc + k_ordera) {
			const dimensions<k_orderb> &dims = bisb.get_dims();
			i1[idimc] = dims[iconn - k_orderc - k_ordera] - 1;
		} else {
			const dimensions<k_ordera> &dims = bisa.get_dims();
			i1[idimc] = dims[iconn - k_orderc] - 1;
		}
	}
	block_index_space<k_orderc> bis(dimensions<k_orderc>(
		index_range<k_orderc>(i0, i1)));
	mask<k_orderc> msk_done;
	for(size_t idimc = 0; idimc < k_orderc; idimc++) {
		if(msk_done[idimc]) continue;
		mask<k_orderc> msk_todo;
		if(conn[idimc] >= k_orderc + k_ordera) {
			size_t type = bisb.get_type(
				conn[idimc] - k_orderc - k_ordera);
			for(size_t idimb = 0; idimb < k_orderb; idimb++) {
				size_t iconn = k_orderc + k_ordera + idimb;
				if(bisb.get_type(idimb) == type &&
					conn[iconn] < k_orderc) {
					msk_todo[conn[iconn]] = true;
				}
			}
			const split_points &pts = bisb.get_splits(type);
			size_t npts = pts.get_num_points();
			for(size_t ipt = 0; ipt < npts; ipt++)
				bis.split(msk_todo, pts[ipt]);
		} else {
			size_t type = bisa.get_type(conn[idimc] - k_orderc);
			for(size_t idima = 0; idima < k_ordera; idima++) {
				size_t iconn = k_orderc + idima;
				if(bisa.get_type(idima) == type &&
					conn[iconn] < k_orderc) {
					msk_todo[conn[iconn]] = true;
				}
			}
			const split_points &pts = bisa.get_splits(type);
			size_t npts = pts.get_num_points();
			for(size_t ipt = 0; ipt < npts; ipt++)
				bis.split(msk_todo, pts[ipt]);
		}
		msk_done |= msk_todo;
	}

	bis.match_splits();
	return bis;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_symmetry() {

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	const block_index_space<k_ordera> &bisa = m_bta.get_bis();
	const block_index_space<k_orderb> &bisb = m_btb.get_bis();

	sequence<N, size_t> mapa(0);
	sequence<M, size_t> mapb(0);

	mask<k_ordera> ma;
	mask<k_orderb> mb;

	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	for(size_t i = 0, j = 0; i < k_ordera; i++) {
		if(conn[k_orderc + i] < k_orderc) {
			ma[i] = true;
			mapa[j++] = i;
		}
	}
	for(size_t i = 0, j = 0; i < k_orderb; i++) {
		if(conn[k_orderc + k_ordera + i] < k_orderc) {
			mb[i] = true;
			mapb[j++] = i;
		}
	}

	//
	//	Build projected block index spaces
	//
	index<N> i1a, i2a;
	index<M> i1b, i2b;
	for(size_t i = 0; i < N; i++) {
		i2a[i] = bisa.get_dims().get_dim(mapa[i]) - 1;
	}
	for(size_t i = 0; i < M; i++) {
		i2b[i] = bisb.get_dims().get_dim(mapb[i]) - 1;
	}
	dimensions<N> rdimsa(index_range<N>(i1a, i2a));
	dimensions<M> rdimsb(index_range<M>(i1b, i2b));
	block_index_space<N> rbisa(rdimsa);
	block_index_space<M> rbisb(rdimsb);

	//
	//	Transfer splits
	//
	mask<N> rma_todo;
	mask<M> rmb_todo;
	while(true) {
		size_t i = 0;
		while(i < N && rma_todo[i] == true) i++;
		if(i == N) break;
		size_t typ = bisa.get_type(mapa[i]);
		mask<N> split_mask;
		for(size_t j = i; j < N; j++) {
			split_mask[j] = bisa.get_type(mapa[j]) == typ;
		}
		const split_points &pts = bisa.get_splits(typ);
		size_t npts = pts.get_num_points();
		for(size_t ipt = 0; ipt < npts; ipt++)
			rbisa.split(split_mask, pts[ipt]);
		rma_todo |= split_mask;
	}
	while(true) {
		size_t i = 0;
		while(i < M && rmb_todo[i] == true) i++;
		if(i == M) break;
		size_t typ = bisb.get_type(mapb[i]);
		mask<M> split_mask;
		for(size_t j = i; j < M; j++) {
			split_mask[j] = bisb.get_type(mapb[j]) == typ;
		}
		const split_points &pts = bisb.get_splits(typ);
		size_t npts = pts.get_num_points();
		for(size_t ipt = 0; ipt < npts; ipt++)
			rbisb.split(split_mask, pts[ipt]);
		rmb_todo |= split_mask;
	}

	symmetry<N, double> rsyma(rbisa);
	symmetry<M, double> rsymb(rbisb);

	so_proj_down<N + K, K, double>(ca.req_const_symmetry(), ma).
		perform(rsyma);
	so_proj_down<M + K, K, double>(cb.req_const_symmetry(), mb).
		perform(rsymb);

	mask<k_orderc> xma;
	mask<k_orderc> xmb;
	sequence<N, size_t> xseqa1(0), xseqa2(0);
	sequence<M, size_t> xseqb1(0), xseqb2(0);
	for(size_t i = k_orderc, j = 0; i < k_orderc + k_ordera; i++) {
		if(conn[i] < k_orderc) xseqa1[j++] = i;
	}
	for(size_t i = k_orderc + k_ordera, j = 0;
		i < k_orderc + k_ordera + k_orderb; i++) {
		if(conn[i] < k_orderc) xseqb1[j++] = i;
	}
	for(size_t i = 0, ja = 0, jb = 0; i < k_orderc; i++) {
		if(conn[i] < k_orderc + k_ordera) {
			xma[i] = true;
			xseqa2[ja++] = conn[i];
		} else {
			xmb[i] = true;
			xseqb2[jb++] = conn[i];
		}
	}
	permutation_builder<N> xpba(xseqa2, xseqa1);
	permutation_builder<M> xpbb(xseqb2, xseqb1);
	symmetry<k_orderc, double> xsyma(m_bis);
	symmetry<k_orderc, double> xsymb(m_bis);
	so_proj_up<N, M, double>(rsyma, xpba.get_perm(), xma).perform(xsyma);
	so_proj_up<M, N, double>(rsymb, xpbb.get_perm(), xmb).perform(xsymb);
	so_union<k_orderc, double>(xsyma, xsymb).perform(m_sym);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule() {

	btod_contract2<N, M, K>::start_timer("make_schedule");

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
	orbit_list<k_orderb, double> olb(cb.req_const_symmetry());
	orbit_list<k_orderc, double> olc(m_sym);

	for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		if(ca.req_is_zero_block(ola.get_index(ioa))) continue;

		orbit<k_ordera, double> oa(ca.req_const_symmetry(),
			ola.get_index(ioa));

		for(typename orbit_list<k_orderb, double>::iterator iob =
			olb.begin(); iob != olb.end(); iob++) {

			if(cb.req_is_zero_block(olb.get_index(iob))) continue;

			orbit<k_orderb, double> ob(cb.req_const_symmetry(),
				olb.get_index(iob));

				make_schedule(oa, ob, olc);
		}
	}

	btod_contract2<N, M, K>::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule(const orbit<k_ordera, double> &oa,
	const orbit<k_orderb, double> &ob,
	const orbit_list<k_orderc, double> &olc) {

	size_t ka[K], kb[K];
	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	for(size_t i = 0, j = 0; i < k_ordera; i++) {
		if(conn[k_orderc + i] >= k_orderc + k_ordera) {
			ka[j] = i;
			kb[j] = conn[k_orderc + i] - k_orderc - k_ordera;
			j++;
		}
	}

	typedef std::multimap<size_t, block_contr_t> local_schedule_t;
	local_schedule_t local_sch;

	for(typename orbit<k_ordera, double>::iterator ia = oa.begin();
		ia != oa.end(); ia++) {

	abs_index<k_ordera> aidxa(oa.get_abs_index(ia), m_bidimsa);
	const index<k_ordera> &idxa = aidxa.get_index();
	const transf<k_ordera, double> &tra = oa.get_transf(
		aidxa.get_abs_index());

	for(typename orbit<k_orderb, double>::iterator ib = ob.begin();
		ib != ob.end(); ib++) {

		abs_index<k_orderb> aidxb(ob.get_abs_index(ib), m_bidimsb);
		const index<k_orderb> &idxb = aidxb.get_index();
		const transf<k_orderb, double> &trb = ob.get_transf(
			aidxb.get_abs_index());

		bool need_contr = true;
		for(size_t i = 0; i < K; i++) {
			if(idxa[ka[i]] != idxb[kb[i]]) {
				need_contr = false;
				break;
			}
		}
		if(!need_contr) continue;

		index<k_orderc> idxc;
		for(size_t i = 0; i < k_orderc; i++) {
			register size_t j = conn[i] - k_orderc;
			idxc[i] = j < k_ordera ? idxa[j] : idxb[j - k_ordera];
		}
		abs_index<k_orderc> aidxc(idxc, m_bidimsc);

		if(!olc.contains(aidxc.get_abs_index())) continue;

		std::pair<typename local_schedule_t::iterator,
			typename local_schedule_t::iterator> itpair =
				local_sch.equal_range(aidxc.get_abs_index());
		bool done = false;
		for(typename local_schedule_t::iterator isch = itpair.first;
			isch != itpair.second; isch++) {

			block_contr_t &bc = isch->second;
			if(bc.is_same_perm(tra, trb)) {
				bc.m_c += tra.get_coeff() * trb.get_coeff();
				done = true;
				break;
			}
		}
		if(!done) {
			block_contr_t bc(oa.get_abs_canonical_index(),
				ob.get_abs_canonical_index(),
				tra.get_coeff() * trb.get_coeff(),
				tra.get_perm(), trb.get_perm());
			local_sch.insert(std::pair<size_t, block_contr_t>(
				aidxc.get_abs_index(), bc));
		}

	} // for ib

	} // for ia

	typename local_schedule_t::iterator ilocsch = local_sch.begin();
	for(; ilocsch != local_sch.end(); ilocsch++) {

		block_contr_t &bc = ilocsch->second;
		if(bc.m_c == 0.0) continue;
		typename schedule_t::iterator isch =
			m_contr_sch.find(ilocsch->first);
		if(isch == m_contr_sch.end()) {
			block_contr_list_t *lst = new block_contr_list_t;
			lst->push_back(bc);
			m_contr_sch.insert(std::pair<size_t, block_contr_list_t*>(
				ilocsch->first, lst));
		} else {
			isch->second->push_back(bc);
		}
		if(!m_sch.contains(ilocsch->first)) {
			m_sch.insert(ilocsch->first);
		}
	}

}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::clear_schedule(schedule_t &sch) {

	typename schedule_t::iterator isch = sch.begin();
	for(; isch != sch.end(); isch++) {
		delete isch->second;
		isch->second = NULL;
	}
	sch.clear();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::contract_block(
	block_contr_list_t &lst, const index<k_orderc> &idxc,
	block_tensor_ctrl<k_ordera, double> &ca,
	block_tensor_ctrl<k_orderb, double> &cb,
	tensor_i<k_orderc, double> &tc, const transf<k_orderc, double> &trc,
	bool zero, double c) {

	if(zero) tod_set<k_orderc>().perform(tc);

	std::list< index<k_ordera> > blksa;
	std::list< index<k_orderb> > blksb;
	std::list< tod_contract2<N, M, K>* > op_ptrs;
	tod_sum<k_orderc> *op_sum = 0;

	for(typename block_contr_list_t::iterator ilst = lst.begin();
		ilst != lst.end(); ilst++) {

		abs_index<k_ordera> aia(ilst->m_absidxa, m_bidimsa);
		abs_index<k_orderb> aib(ilst->m_absidxb, m_bidimsb);
		const index<k_ordera> &ia = aia.get_index();
		const index<k_orderb> &ib = aib.get_index();

		bool zeroa = ca.req_is_zero_block(ia);
		bool zerob = cb.req_is_zero_block(ib);
		if(zeroa || zerob) continue;

		tensor_i<k_ordera, double> &blka = ca.req_block(ia);
		tensor_i<k_orderb, double> &blkb = cb.req_block(ib);
		blksa.push_back(ia);
		blksb.push_back(ib);

		contraction2<N, M, K> contr(m_contr);
		contr.permute_a(ilst->m_perma);
		contr.permute_b(ilst->m_permb);
		contr.permute_c(trc.get_perm());

		tod_contract2<N, M, K> *controp =
			new tod_contract2<N, M, K>(contr, blka, blkb);
		double kc = ilst->m_c * trc.get_coeff();
		op_ptrs.push_back(controp);
		if(op_sum == 0) op_sum = new tod_sum<k_orderc>(*controp, kc);
		else op_sum->add_op(*controp, kc);
	}

	if(op_sum != 0) {
		op_sum->prefetch();
		op_sum->perform(tc, c);
		delete op_sum; op_sum = 0;
		for(typename std::list< tod_contract2<N, M, K>* >::const_iterator iptr =
			op_ptrs.begin(); iptr != op_ptrs.end(); iptr++) {

			delete *iptr;
		}
		op_ptrs.clear();
	}

	for(typename std::list< index<k_ordera> >::const_iterator i =
		blksa.begin(); i != blksa.end(); i++) ca.ret_block(*i);
	for(typename std::list< index<k_orderb> >::const_iterator i =
		blksb.begin(); i != blksb.end(); i++) cb.ret_block(*i);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
