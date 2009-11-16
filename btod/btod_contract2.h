#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include <list>
#include <map>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "symmetry/so_projdown.h"
#include "symmetry/so_projup.h"
#include "tod/contraction2.h"
#include "tod/tod_contract2.h"
#include "tod/tod_set.h"
#include "btod_additive.h"
#include "btod_so_copy.h"

namespace libtensor {


/**	\brief Contraction of two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 :
	public btod_additive<N + M>,
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
	block_tensor_i<k_ordera, double> &m_bta; //!< First argument (a)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (b)
	block_index_space<k_orderc> m_bis; //!< Block %index space of the result
	symmetry<k_orderc, double> m_sym; //!< Symmetry of the result

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation
		\param contr Contraction.
		\param bta Block %tensor a (first argument).
		\param btb Block %tensor b (second argument).
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
	virtual const block_index_space<N + M> &get_bis() const;
	virtual const symmetry<N + M, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<k_orderc, double> &btc)
		throw(exception);
	virtual void perform(block_tensor_i<k_orderc, double> &btc,
		const index<k_orderc> &idx) throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N + M>
	//@{
	virtual void perform(block_tensor_i<k_orderc, double> &btc, double c)
		throw(exception);
	//@}

private:
	static block_index_space<N + M> make_bis(
		const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);
	void make_symmetry();

	void do_perform(block_tensor_i<k_orderc, double> &btc, bool zero,
		double c) throw(exception);

	/**	\brief For an orbit in a and b, make a list of blocks in c
	 **/
	void make_schedule(
		schedule_t &sch, const dimensions<k_ordera> &bidimsa,
		const orbit<k_ordera, double> &orba,
		const dimensions<k_orderb> &bidimsb,
		const orbit<k_orderb, double> &orbb,
		const dimensions<k_orderc> &bidimsc,
		const orbit_list<k_orderc, double> &orblstc);

	void clear_schedule(schedule_t &sch);

	void contract_block(
		block_contr_list_t &lst, const index<k_orderc> &idxc,
		block_tensor_ctrl<k_ordera, double> &ctrla,
		const dimensions<k_ordera> &bidimsa,
		block_tensor_ctrl<k_orderb, double> &ctrlb,
		const dimensions<k_orderb> &bidimsb,
		block_tensor_ctrl<k_orderc, double> &ctrlc,
		const dimensions<k_orderc> &bidimsc,
		bool zero, double c);

private:
	btod_contract2<N, M, K> &operator=(const btod_contract2<N, M, K>&);

};


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb)
: m_contr(contr), m_bta(bta), m_btb(btb), m_bis(make_bis(contr, bta, btb)),
	m_sym(m_bis) {

	make_symmetry();
}


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::~btod_contract2() {

}


template<size_t N, size_t M, size_t K>
inline const block_index_space<N + M> &btod_contract2<N, M, K>::get_bis()
	const {

	return m_bis;
}


template<size_t N, size_t M, size_t K>
const symmetry<N + M, double> &btod_contract2<N, M, K>::get_symmetry() const {

	return m_sym;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<k_orderc, double> &btc,
	double c) throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N + M, double>&, double)";

	if(!m_bis.equals(btc.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	btod_contract2<N, M, K>::start_timer();

	block_tensor_ctrl<k_orderc, double> ctrl_btc(btc);

	symmetry<k_orderc, double> sym(m_sym);

	if(sym.equals(ctrl_btc.req_symmetry())) {
		// A*B and C have the same symmetry
		do_perform(btc, false, c);
	} else {
		sym.set_intersection(ctrl_btc.req_symmetry());
		if(sym.equals(m_sym)) {
			// C has a higher symmetry
			throw_exc(k_clazz, method, "Case 1 not handled.");
		} else if(sym.equals(ctrl_btc.req_symmetry())) {
			// A*B has a higher symmetry
			throw_exc(k_clazz, method, "Case 2 not handled.");
		} else {
			throw_exc(k_clazz, method, "Case 3 not handled.");
		}
	}

	btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<k_orderc, double> &btc,
	const index<k_orderc> &idx) throw(exception) {

	throw_exc(k_clazz, "perform(const index<N + M>&)", "NIY");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<k_orderc, double> &btc)
	throw(exception) {

	static const char *method = "perform(block_tensor_i<N + M, double>&)";

	block_index_space<k_orderc> bisc(btc.get_bis());
	bisc.match_splits();
	if(!m_bis.equals(bisc)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	btod_contract2<N, M, K>::start_timer();

	btod_so_copy<k_orderc> symcopy(m_sym);
	symcopy.perform(btc);

	do_perform(btc, true, 1.0);

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


namespace btod_contract2_ns {

template<size_t N, size_t M, size_t K, size_t L>
class projector {
private:
	const sequence<2 * (N + M + K), size_t> &m_conn;
	block_tensor_i<N + K, double> &m_bta;
	const block_index_space<N + M> &m_bisc;
	symmetry<N + M, double> &m_symc;

public:
	projector(const sequence<2 * (N + M + K), size_t> &conn,
		block_tensor_i<N + K, double> &bta,
		const block_index_space<N + M> &bisc,
		symmetry<N + M, double> &symc) :
		m_conn(conn), m_bta(bta), m_bisc(bisc), m_symc(symc) { }
	void project();
};


template<size_t M, size_t K, size_t L>
class projector<0, M, K, L> {
public:
	projector(const sequence<2 * (M + K), size_t> &conn,
		block_tensor_i<K, double> &bta,
		const block_index_space<M> &bisc,
		symmetry<M, double> &symc) { }
	void project() { }
};


template<size_t N, size_t M, size_t K, size_t L>
void projector<N, M, K, L>::project() {

	dimensions<N + K> bidimsa(m_bta.get_bis().get_block_index_dims());
	dimensions<N + M> bidimsc(m_bisc.get_block_index_dims());

	index<N> ia1, ia2;
	mask<N + K> projmska;
	mask<N + M> projmskca;
	size_t j = 0;
	for(size_t i = 0; i < N + K; i++) {
		size_t iconn = m_conn[N + M + L + i];
		if(iconn < N + M) {
			ia2[j] = bidimsa[i] - 1;
			projmska[i] = true;
			projmskca[iconn] = true;
		}
	}
	dimensions<N> projdimsa(index_range<N>(ia1, ia2));
	block_tensor_ctrl<N + K, double> ctrla(m_bta);
	const symmetry<N + K, double> &syma = ctrla.req_symmetry();
	typename symmetry<N + K, double>::iterator ielema = syma.begin();
	for(; ielema != syma.end(); ielema++) {
		so_projdown<N + K, K, double> projdn(
			syma.get_element(ielema), projmska, projdimsa);
		if(!projdn.is_identity()) {
			so_projup<N, M, double> projup(
				projdn.get_proj(), projmskca, bidimsc);
			m_symc.add_element(projup.get_proj());
		}
	}
}


}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_symmetry() {

	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	btod_contract2_ns::projector<N, M, K, 0>(
		conn, m_bta, m_bis, m_sym).project();
	btod_contract2_ns::projector<M, N, K, N + K>(
		conn, m_btb, m_bis, m_sym).project();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::do_perform(
	block_tensor_i<k_orderc, double> &btc, bool zero, double c)
	throw(exception) {

	block_tensor_ctrl<k_orderc, double> ctrl_btc(btc);
	block_tensor_ctrl<k_ordera, double> ctrl_bta(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrl_btb(m_btb);

	dimensions<k_ordera> bidimsa(m_bta.get_bis().get_block_index_dims());
	dimensions<k_orderb> bidimsb(m_btb.get_bis().get_block_index_dims());
	dimensions<k_orderc> bidimsc(btc.get_bis().get_block_index_dims());

	//	Go over orbits in A and B and create the schedule

	schedule_t sch;

	orbit_list<k_ordera, double> orblsta(ctrl_bta.req_symmetry());
	orbit_list<k_orderb, double> orblstb(ctrl_btb.req_symmetry());
	orbit_list<k_orderc, double> orblstc(ctrl_btc.req_symmetry());
	typename orbit_list<k_ordera, double>::iterator iorba = orblsta.begin();
	for(; iorba != orblsta.end(); iorba++) {
		orbit<k_ordera, double> orba(ctrl_bta.req_symmetry(),
			orblsta.get_index(iorba));
		typename orbit_list<k_orderb, double>::iterator iorbb =
			orblstb.begin();
		for(; iorbb != orblstb.end(); iorbb++) {
			orbit<k_orderb, double> orbb(ctrl_btb.req_symmetry(),
				orblstb.get_index(iorbb));
			make_schedule(sch, bidimsa, orba, bidimsb, orbb,
				bidimsc, orblstc);
		}
	}

	//	Invoke contractions

	try {
		index<k_orderc> idxc;
		typename schedule_t::iterator isch = sch.begin();
		for(; isch != sch.end(); isch++) {
			bidimsc.abs_index(isch->first, idxc);
			contract_block(*isch->second, idxc, ctrl_bta, bidimsa,
				ctrl_btb, bidimsb, ctrl_btc, bidimsc, zero, c);
		}
	} catch(...) {
		clear_schedule(sch);
		throw;
	}
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule(
	schedule_t &sch, const dimensions<k_ordera> &bidimsa,
	const orbit<k_ordera, double> &orba,
	const dimensions<k_orderb> &bidimsb,
	const orbit<k_orderb, double> &orbb,
	const dimensions<k_orderc> &bidimsc,
	const orbit_list<k_orderc, double> &orblstc) {

	btod_contract2<N, M, K>::start_timer("make_schedule");

	typedef std::multimap<size_t, block_contr_t> local_schedule_t;
	local_schedule_t local_sch;

	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	index<k_ordera> idxa;
	index<k_orderb> idxb;
	index<k_orderc> idxc;

	typename orbit<k_ordera, double>::iterator iidxa = orba.begin();
	for(; iidxa != orba.end(); iidxa++) {
		bidimsa.abs_index(orba.get_abs_index(iidxa), idxa);
		const transf<k_ordera, double> &transfa =
			orba.get_transf(iidxa);

		typename orbit<k_orderb, double>::iterator iidxb =
			orbb.begin();
		for(; iidxb != orbb.end(); iidxb++) {
			bidimsb.abs_index(orbb.get_abs_index(iidxb), idxb);
			const transf<k_orderb, double> &transfb =
				orbb.get_transf(iidxb);

			bool need_contr = true;
			for(size_t i = 0; i < k_ordera; i++) {
				register size_t iconn = conn[k_orderc + i];
				if(iconn < k_orderc) {
					idxc[iconn] = idxa[i];
				} else {
					iconn -= k_orderc + k_ordera;
					if(idxa[i] != idxb[iconn]) {
						need_contr = false;
						break;
					}
				}
			}
			if(!need_contr) continue;
			for(size_t i = 0; i < k_orderb; i++) {
				register size_t iconn =
					conn[k_orderc + k_ordera + i];
				if(iconn < k_orderc) {
					idxc[iconn] = idxb[i];
				}
			}

			size_t absidxc = bidimsc.abs_index(idxc);
			if(!orblstc.contains(absidxc)) continue;

			std::pair<typename local_schedule_t::iterator,
				typename local_schedule_t::iterator> itpair =
					local_sch.equal_range(absidxc);
			bool done = false;
			typename local_schedule_t::iterator isch = itpair.first;
			for(; isch != itpair.second; isch++) {
				block_contr_t &bc = isch->second;
				if(bc.is_same_perm(transfa, transfb)) {
					bc.m_c += transfa.get_coeff() *
						transfb.get_coeff();
					done = true;
					break;
				}
			}
			if(!done) {
				block_contr_t bc(
					orba.get_abs_canonical_index(),
					orbb.get_abs_canonical_index(),
					transfa.get_coeff() *
						transfb.get_coeff(),
					transfa.get_perm(),
					transfb.get_perm());
				local_sch.insert(std::pair<size_t,
					block_contr_t>(absidxc, bc));
			}

		}
	}

	typename local_schedule_t::iterator ilocsch = local_sch.begin();
	for(; ilocsch != local_sch.end(); ilocsch++) {
		block_contr_t &bc = ilocsch->second;
		if(bc.m_c == 0.0) continue;
		typename schedule_t::iterator isch = sch.find(ilocsch->first);
		if(isch == sch.end()) {
			block_contr_list_t *lst = new block_contr_list_t;
			lst->push_back(bc);
			sch.insert(std::pair<size_t, block_contr_list_t*>(
				ilocsch->first, lst));
		} else {
			isch->second->push_back(bc);
		}
	}

	btod_contract2<N, M, K>::stop_timer("make_schedule");

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
	block_tensor_ctrl<k_ordera, double> &ctrla,
	const dimensions<k_ordera> &bidimsa,
	block_tensor_ctrl<k_orderb, double> &ctrlb,
	const dimensions<k_orderb> &bidimsb,
	block_tensor_ctrl<k_orderc, double> &ctrlc,
	const dimensions<k_orderc> &bidimsc,
	bool zero, double c) {

	index<k_ordera> idxa;
	index<k_orderb> idxb;

	bool adjzero = zero || ctrlc.req_is_zero_block(idxc);
	tensor_i<k_orderc, double> &tc = ctrlc.req_block(idxc);

	if(adjzero) tod_set<k_orderc>().perform(tc);

	typename block_contr_list_t::iterator ilst = lst.begin();
	for(; ilst != lst.end(); ilst++) {
		bidimsa.abs_index(ilst->m_absidxa, idxa);
		bidimsb.abs_index(ilst->m_absidxb, idxb);
		if(ctrla.req_is_zero_block(idxa) ||
			ctrlb.req_is_zero_block(idxb)) continue;

		tensor_i<k_ordera, double> &ta = ctrla.req_block(idxa);
		tensor_i<k_orderb, double> &tb = ctrlb.req_block(idxb);

		contraction2<N, M, K> contr(m_contr);
		contr.permute_a(ilst->m_perma);
		contr.permute_b(ilst->m_permb);
		tod_contract2<N, M, K> controp(contr, ta, tb);
		controp.perform(tc, c * ilst->m_c);

		ctrla.ret_block(idxa);
		ctrlb.ret_block(idxb);
	}

	ctrlc.ret_block(idxc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
