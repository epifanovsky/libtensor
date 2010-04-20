#ifndef LIBTENSOR_ADDITION_SCHEDULE_H
#define LIBTENSOR_ADDITION_SCHEDULE_H

#include <list>
#include <vector>
#include "../core/abs_index.h"
#include "../core/block_index_space.h"
#include "../core/symmetry.h"
#include "../symmetry/so_add.h"
#include "assignment_schedule.h"

namespace libtensor {


/**	\brief Creates a schedule for the blockwise addition of two block
		tensors
	\tparam N Tensor order.

	Given the block index space and symmetries of two block tensors,
	this routine creates a schedule for the blockwise addition. The schedule
	is sorted such that the blocks of the first operand are accessed only
	once.

	Block %tensor operations shall use this routine to implement their
	additive interface.

	\ingroup libtensor_btod
 **/
template<size_t N, typename T>
class addition_schedule {
public:
	typedef struct tier2_node_ {
		size_t cib; //!< Canonical %index in B
		transf<N, T> trb; //!< Transformation of B
		tier2_node_(size_t cib_, const transf<N, T> &trb_) :
			cib(cib_), trb(trb_) { }
	} tier2_node_t;

	typedef struct tier3_node_ {
		size_t cib; //!< Canonical %index in B
		transf<N, T> tra; //!< Transformation of A
		tier3_node_(size_t cib_, const transf<N, T> &tra_) :
			cib(cib_), tra(tra_) { }
	} tier3_node_t;

	typedef struct tier4_node_ {
		size_t cib; //!< Canonical %index in B
		size_t cic; //!< Canonical %index in C
		transf<N, T> tra; //!< Transformation of A
		transf<N, T> trb; //!< Transformation of B
		tier4_node_(size_t cib_, size_t cic_, const transf<N, T> &tra_,
			const transf<N, T> &trb_) :
			cib(cib_), cic(cic_), tra(tra_), trb(trb_) { }
	} tier4_node_t;

	typedef std::list<tier3_node_t> tier3_list_t;
	typedef std::list<tier4_node_t> tier4_list_t;

	typedef struct schedule_node_ {
		size_t cia; //!< Canonical %index in A
		bool tier1; //!< Tier 1 marker (for consistency check only)
		tier2_node_t *tier2; //!< Tier 2 node
		tier3_list_t *tier3; //!< Tier 3 list
		tier4_list_t *tier4; //!< Tier 4 list

		schedule_node_(size_t cia_) : cia(cia_), tier1(false), tier2(0),
			tier3(0), tier4(0) { }
	} schedule_node_t;

	typedef std::list<schedule_node_t> schedule_t; //!< Schedule type

	typedef typename schedule_t::const_iterator iterator;

private:
	const symmetry<N, T> &m_syma; //!< Symmetry of A
	const symmetry<N, T> &m_symb; //!< Symmetry of B
	symmetry<N, T> m_symc; //!< Largest common subgroup of A and B
	schedule_t m_sch; // Additive schedule

public:
	/**	\brief Initializes the algorithm
	 **/
	addition_schedule(const symmetry<N, T> &syma,
		const symmetry<N, T> &symb);

	/**	\brief Destructor
	 **/
	~addition_schedule();

	/**	\brief Runs the algorithm
	 **/
	void build(const assignment_schedule<N, T> &asch);

	iterator begin() const {
		return m_sch.begin();
	}

	iterator end() const {
		return m_sch.end();
	}

	const schedule_node_t &get_node(const iterator &i) const {
		return *i;
	}

private:
	/**	\brief Removes all elements from the schedule
	 **/
	void clean_schedule() throw();

	/**	\brief Puts 2 in the positions corresponding to canonical
			indexes and 1 for non-canonical indexes
	 **/
	void mark_orbits(const symmetry<N, T> &sym, std::vector<char> &o);

	/**	\brief Recursive part of mark_orbits()
	 **/
	void mark_orbit(const symmetry<N, T> &sym, const abs_index<N> &aci,
		std::vector<char> &o);

	/**	\brief Returns the canonical %index and a transformation
			to a given %index
	 **/
	size_t find_canonical(const dimensions<N> &bidims,
		const symmetry<N, T> &sym, const abs_index<N> ai,
		transf<N, T> &tr, const std::vector<char> &o);

	size_t find_canonical_iterate(const dimensions<N> &bidims,
		const symmetry<N, T> &sym, const abs_index<N> &ai,
		transf<N, T> &tr, const std::vector<char> &o,
		std::vector<char> &o2);

	/**	\brief Processes one orbit in A and marks it
		\param acia Absolute canonical index in A.
		\param oa Array of visited indexes.
	 **/
	void process_orbit_in_a(const dimensions<N> &bidims,
		const abs_index<N> &acia, const abs_index<N> &aia,
		const transf<N, T> &tra,
		std::vector<char> &oa, const std::vector<char> &ob,
		const std::vector<char> &oc, schedule_node_t &node);

	void iterate_sym_elements_in_a(const dimensions<N> &bidims,
		const abs_index<N> &acia,
		const abs_index<N> &aia, const transf<N, T> &tra,
		std::vector<char> &oa, const std::vector<char> &ob,
		const std::vector<char> &oc, schedule_node_t &node);

private:
	addition_schedule(const addition_schedule<N, T>&);
	const addition_schedule<N, T> &operator=(
		const addition_schedule<N, T>&);
};


template<size_t N, typename T>
addition_schedule<N, T>::addition_schedule(const symmetry<N, T> &syma,
	const symmetry<N, T> &symb) :

	m_syma(syma), m_symb(symb), m_symc(m_symb.get_bis()) {

	permutation<N> perm0;
	so_add<N, T>(m_syma, perm0, m_symb, perm0).perform(m_symc);
}


template<size_t N, typename T>
addition_schedule<N, T>::~addition_schedule() {

	clean_schedule();
}


template<size_t N, typename T>
void addition_schedule<N, T>::build(const assignment_schedule<N, T> &asch) {

	//
	//	For each allowed orbit Oa in A:
	//	  For each index Ja:(Ia, Ta) in Oa:
	//	    Jc := P Ja;
	//	    If Jc is not canonical in C, continue;
	//	    If Jc is canonical in B:
	//	      Add to Tier 2: Jc <- Jc + (Ia, Ta)
	//	    Else:
	//	      Find canonical index in B: Jc:(Ib, Tb)
	//	      Add to Tier 1: Jc <- (Ib, Tb) + (Ia, Ta)
	//	    End If
	//	  End
	//	End
	//
	
	clean_schedule();
	dimensions<N> bidims(m_syma.get_bis().get_block_index_dims());

	std::vector<char> oa(bidims.get_size(), 0),
		ob(bidims.get_size(), 0), oc(bidims.get_size(), 0);
	mark_orbits(m_symb, ob);
	mark_orbits(m_symc, oc);

	for(typename assignment_schedule<N, T>::iterator i = asch.begin();
		i != asch.end(); i++) {

		abs_index<N> acia(asch.get_abs_index(i), bidims);
		transf<N, T> tra0;
		if(oa[acia.get_abs_index()] == 0) {
			schedule_node_t node(acia.get_abs_index());
			process_orbit_in_a(bidims, acia, acia, tra0, oa, ob,
				oc, node);
			m_sch.push_back(node);
		}
	}
}


template<size_t N, typename T>
void addition_schedule<N, T>::clean_schedule() throw() {

	for(typename schedule_t::iterator i = m_sch.begin(); i != m_sch.end();
		i++) {

		delete i->tier2;
		delete i->tier3;
		delete i->tier4;
	}
	m_sch.clear();
}


template<size_t N, typename T>
void addition_schedule<N, T>::mark_orbits(const symmetry<N, T> &sym,
	std::vector<char> &o) {

	dimensions<N> bidims(sym.get_bis().get_block_index_dims());
	abs_index<N> aci(bidims);
	do {
		if(o[aci.get_abs_index()] == 0) {
			o[aci.get_abs_index()] = 2;
			mark_orbit(sym, aci, o);
		}
	} while(aci.inc());
}


template<size_t N, typename T>
void addition_schedule<N, T>::mark_orbit(const symmetry<N, T> &sym,
	const abs_index<N> &ai, std::vector<char> &o) {

	for(typename symmetry<N, T>::iterator is = sym.begin();
		is != sym.end(); is++) {

		const symmetry_element_set<N, T> &es = sym.get_subset(is);

		for(typename symmetry_element_set<N, T>::const_iterator ie =
			es.begin(); ie != es.end(); ie++) {

			const symmetry_element_i<N, T> &e = es.get_elem(ie);
			index<N> i1(ai.get_index());
			e.apply(i1);
			abs_index<N> ai1(i1, ai.get_dims());
			if(o[ai1.get_abs_index()] == 0) {
				o[ai1.get_abs_index()] = 1;
				mark_orbit(sym, ai1, o);
			}
		}
	}
}


template<size_t N, typename T>
size_t addition_schedule<N, T>::find_canonical(const dimensions<N> &bidims,
	const symmetry<N, T> &sym, const abs_index<N> ai, transf<N, T> &tr,
	const std::vector<char> &o) {

	if(o[ai.get_abs_index()] == 2) return ai.get_abs_index();

	std::vector<char> o2(o.size(), 0);

	transf<N, T> tr1; // From current to canonical
	size_t ii = find_canonical_iterate(bidims, sym, ai, tr1, o, o2);
	tr1.invert(); // From canonical to current
	tr.transform(tr1);
	return ii;
}


template<size_t N, typename T>
size_t addition_schedule<N, T>::find_canonical_iterate(
	const dimensions<N> &bidims, const symmetry<N, T> &sym,
	const abs_index<N> &ai, transf<N, T> &tr,
	const std::vector<char> &o, std::vector<char> &o2) {

	o2[ai.get_abs_index()] = 1;
	size_t smallest = ai.get_abs_index();

	for(typename symmetry<N, T>::iterator is = sym.begin();
		is != sym.end(); is++) {

		const symmetry_element_set<N, T> &es = sym.get_subset(is);

		for(typename symmetry_element_set<N, T>::const_iterator ie =
			es.begin(); ie != es.end(); ie++) {

			const symmetry_element_i<N, T> &e = es.get_elem(ie);
			index<N> i1(ai.get_index());
			transf<N, T> tr1;
			e.apply(i1, tr1);
			abs_index<N> ai1(i1, bidims);
			if(o[ai1.get_abs_index()] == 2) {
				tr.transform(tr1);
				return ai1.get_abs_index();
			}
			if(o2[ai1.get_abs_index()] == 0) {
				size_t ii = find_canonical_iterate(bidims,
					sym, ai1, tr1, o, o2);
				if(o[ii] == 2) {
					tr.transform(tr1);
					return ii;
				}
				if(ii < smallest) smallest = ii;
			}
		}
	}

	return smallest;
}


template<size_t N, typename T>
void addition_schedule<N, T>::process_orbit_in_a(const dimensions<N> &bidims,
	const abs_index<N> &acia,
	const abs_index<N> &aia, const transf<N, T> &tra,
	std::vector<char> &oa,
	const std::vector<char> &ob, const std::vector<char> &oc,
	schedule_node_t &node) {

	oa[aia.get_abs_index()] = 1;

	//
	//	Index in B and C that corresponds to the index in A
	//
	index<N> ib(aia.get_index());
	abs_index<N> aib(ib, bidims);

	//
	//	Skip all non-canonical blocks in C
	//
	if(oc[aib.get_abs_index()] == 2) {

		bool cana = aia.get_abs_index() == acia.get_abs_index();
		bool canb = ob[aib.get_abs_index()] == 2;

		//
		//	Tier 1: Canonical in A, B and C
		//
		if(cana && canb) {

			//~ std::cout << "tier 1: "
				//~ << "C" << aib.get_index() << " <- "
				//~ << "B" << aib.get_index() << " + "
				//~ << "A" << aia.get_index() << std::endl;
			node.tier1 = true;
			if(node.tier2 != 0) {
				// throw bad_symmetry();
			}

		}

		//
		//	Tier 2: Canonical in A and C
		//
		if(cana && !canb) {

			transf<N, T> trb;
			abs_index<N> acib(find_canonical(bidims, m_symb, aib,
				trb, ob), bidims);

			//~ std::cout << "tier 2: "
				//~ << "C" << aib.get_index() << " <- "
				//~ << "b" << aib.get_index() << "<("
				//~ << trb.get_perm() << ", " << trb.get_coeff()
				//~ << ")-" << acib.get_index() << " + "
				//~ << "A" << aia.get_index() << std::endl;
			if(node.tier1 == true) {
				// throw bad_symmetry
			}
			node.tier2 = new tier2_node_t(
				acib.get_abs_index(), trb);

		}

		//
		//	Tier 3: Canonical in B and C
		//
		if(!cana && canb) {

			//~ std::cout << "tier 3: "
				//~ << "C" << aib.get_index() << " <- "
				//~ << "B" << aib.get_index() << " + "
				//~ << "a" << aia.get_index() << "<("
				//~ << tra.get_perm() << ", " << tra.get_coeff()
				//~ << ")-" << acia.get_index()
				//~ << std::endl;

			if(node.tier3 == 0) node.tier3 = new tier3_list_t;
			node.tier3->push_back(tier3_node_t(
				aib.get_abs_index(), tra));
		}

		//
		//	Tier 4: Canonical only in C
		//
		if(!cana && !canb) {

			transf<N, T> trb;
			abs_index<N> acib(find_canonical(bidims, m_symb, aib,
				trb, ob), bidims);

			//~ std::cout << "tier 4: "
				//~ << "C" << aib.get_index() << " <- "
				//~ << "b" << aib.get_index() << "<("
				//~ << trb.get_perm() << ", " << trb.get_coeff()
				//~ << ")-" << acib.get_index() << " + "
				//~ << "a" << aia.get_index() << "<("
				//~ << tra.get_perm() << ", " << tra.get_coeff()
				//~ << ")-" << acia.get_index()
				//~ << std::endl;

			if(node.tier4 == 0) node.tier4 = new tier4_list_t;
			node.tier4->push_back(tier4_node_t(acib.get_abs_index(),
				aib.get_abs_index(), tra, trb));
		}

	}

	//
	//	Continue exploring the orbit recursively
	//
	iterate_sym_elements_in_a(bidims, acia, aia, tra, oa, ob, oc, node);
}


template<size_t N, typename T>
void addition_schedule<N, T>::iterate_sym_elements_in_a(
	const dimensions<N> &bidims,
	const abs_index<N> &acia, const abs_index<N> &aia,
	const transf<N, T> &tra, std::vector<char> &oa,
	const std::vector<char> &ob, const std::vector<char> &oc,
	schedule_node_t &node) {

	for(typename symmetry<N, T>::iterator is = m_syma.begin();
		is != m_syma.end(); is++) {

		const symmetry_element_set<N, T> &es = m_syma.get_subset(is);

		for(typename symmetry_element_set<N, T>::const_iterator ie =
			es.begin(); ie != es.end(); ie++) {

			const symmetry_element_i<N, T> &e = es.get_elem(ie);
			index<N> ia1(aia.get_index());
			transf<N, T> tra1(tra);
			e.apply(ia1, tra1);
			abs_index<N> aia1(ia1, bidims);
			if(oa[aia1.get_abs_index()] == 0) {
				process_orbit_in_a(bidims, acia, aia1, tra1, oa,
					ob, oc, node);
			}
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITION_SCHEDULE_H
