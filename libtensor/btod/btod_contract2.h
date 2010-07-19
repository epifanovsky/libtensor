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
#include "../tod/contraction2.h"
#include "additive_btod.h"
#include "../not_implemented.h"
#include "bad_block_index_space.h"

namespace libtensor {

template<size_t N, size_t M, size_t K>
class btod_contract2_symmetry_builder_base;

template<size_t N, size_t M, size_t K> class btod_contract2_symmetry_builder;

template<size_t N, size_t K> class btod_contract2_symmetry_builder<N, N, K>;


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
	btod_contract2_symmetry_builder<N, M, K> m_sym_bld;
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
		return m_sym_bld.get_bis();
	}

	virtual const symmetry<N + M, double> &get_symmetry() const {
		return m_sym_bld.get_symmetry();
	}

	virtual const assignment_schedule<N + M, double> &get_schedule() const {
		return m_sch;
	}

	virtual void sync_on();
	virtual void sync_off();

	//@}

	using additive_btod<N + M>::perform;

protected:
	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i);
	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i, const transf<N + M, double> &tr,
		double c);

private:
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
class btod_contract2_symmetry_builder_base {
private:
	block_index_space<N + M + 2 * K> m_xbis;
	block_index_space<N + M> m_bis;
	symmetry<N + M, double> m_sym;

public:
	btod_contract2_symmetry_builder_base(
		const contraction2<N, M, K> &contr,
		const block_index_space<N + M + 2 * K> &xbis) :
		m_xbis(xbis), m_bis(make_bis(contr, m_xbis)), m_sym(m_bis) { }

	const block_index_space<N + M> &get_bis() const {
		return m_bis;
	}

	const symmetry<N + M, double> &get_symmetry() const {
		return m_sym;
	}

protected:
	const block_index_space<N + M + 2 * K> &get_xbis() const {
		return m_xbis;
	}

	symmetry<N + M, double> &get_symmetry() {
		return m_sym;
	}

	void make_symmetry(const contraction2<N, M, K> &contr,
		block_tensor_i<N + K, double> &bta,
		block_tensor_i<M + K, double> &btb);

	static block_index_space<N + M + 2 * K> make_xbis(
		const block_index_space<N + K> &bisa,
		const block_index_space<M + K> &bisb);

	static block_index_space<N + M> make_bis(
		const contraction2<N, M, K> &contr,
		const block_index_space<N + M + 2 * K> &xbis);

};


/**	\brief Builds the %symmetry and block %index space for btod_contract2

	\sa btod_contract2<N, M, K>

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2_symmetry_builder :
	public btod_contract2_symmetry_builder_base<N, M, K> {

public:
	btod_contract2_symmetry_builder(const contraction2<N, M, K> &contr,
		block_tensor_i<N + K, double> &bta,
		block_tensor_i<M + K, double> &btb);

};


/**	\brief Builds the %symmetry and block %index space for btod_contract2
		(specialized for same-order A and B)

	\sa btod_contract2<N, M, K>

	\ingroup libtensor_btod
 **/
template<size_t N, size_t K>
class btod_contract2_symmetry_builder<N, N, K> :
	public btod_contract2_symmetry_builder_base<N, N, K> {

public:
	btod_contract2_symmetry_builder(const contraction2<N, N, K> &contr,
		block_tensor_i<N + K, double> &bta,
		block_tensor_i<N + K, double> &btb);

private:
	static block_index_space<2 * (N + K)> make_xbis(
		const block_index_space<N + K> &bisa);

	void make_symmetry(const contraction2<N, N, K> &contr,
		block_tensor_i<N + K, double> &bta);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_contract2_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BTOD_CONTRACT2_H
