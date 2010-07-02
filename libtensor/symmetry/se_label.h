#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include "../defs.h"
#include "../core/block_index_space.h"
#include "../core/mask.h"
#include "../core/symmetry_element_i.h"
#include "product_table_container.h"

namespace libtensor {

/**	\brief Symmetry element for label assigned to %tensor blocks
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	This %symmetry elements establishes a labeling of blocks along each
	dimension of a block %tensor and a mapping of N labels onto labels of
	%tensor blocks via a product table.

	The resulting %tensor block labels are compared to the target label of the
	symmetry element. Blocks with the same label(s) as the target label are
	accepted, all others are discarded.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_label : public symmetry_element_i<N, T> {
public:
	static const char *k_clazz; //!< Class name
	static const char *k_sym_type; //!< Symmetry type

	typedef product_table_container::id_t table_id_t;
	typedef product_table::label_t label_t;

private:
	typedef std::vector<label_t> label_list;

	dimensions<N> m_bidims; //!< Block %index space dimensions
	sequence<N, size_t> m_type; //!< Label type
	sequence<N, label_list*> m_labels; //!< Block labels
	label_t m_label; //!< Target label

	const product_table_i &m_pt; //!< Product table for labels
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the %symmetry element
		\param bis Block %index space.
		\param table_id Id of product table.
		\param target Target label.
	 **/
	se_label(const block_index_space<N> &bis,
			const char *table_id, const label_t &target);

	/**	\brief Copy constructor
	 **/
	se_label(const se_label<N, T> &elem);

	/**	\brief Virtual destructor
	 **/
	virtual ~se_label();

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Assign label to a subspace block given by a %mask and a position

		\param msk Dimension mask
		\param pos Block position (not to exceed the number of
			splits along the given dimension)
		\param label The label
		\throw bad_parameter If the mask is incorrect.
		\throw out_of_bounds If the position or the label is out of bounds.
	 **/
	void assign(const mask<N> &msk, size_t pos,
			label_t label) throw(bad_parameter, out_of_bounds);

	/**	\brief Remove label of a subspace block given by a %mask and a position

		\param msk Dimension mask
		\param pos Block position (not to exceed the number of
			splits along the given dimension)
		\throw bad_parameter If the mask is incorrect.
		\throw out_of_bounds If the position is out of bounds.
	 **/
	void remove(const mask<N> &msk, size_t pos) throw(bad_parameter,
			out_of_bounds);

	/**	\brief Remove all labels
	 **/
	void clear();

	//@}

	/**	\brief Returns the type (labeling pattern) of a dimension
		\param dim Dimension number.
		\throw out_of_bounds If the dimension number is out of bounds.
	 **/
	size_t get_type(size_t dim) const;

	/**	\brief Returns the block dimension of a dimension type
		\param type Dimension type.
		\throw bad_parameter If the dimension type is invalid.
	 **/
	size_t get_dim(size_t type) const throw(bad_parameter);

	/**	\brief Returns the block dimensions.
	 **/
	const dimensions<N> &get_dims() const {
		return m_bidims;
	}

	/**	\brief Returns the label of a block of a dimension type.
		\param type Dimension type.
		\param pos Block position.
		\throw out_of_bounds If the dimension type is out of bounds.
	 **/
	label_t get_label(size_t type, size_t pos) const throw(out_of_bounds);

	//!	\name Implementation of symmetry_element_i<N, T>
	//@{

	/**	\copydoc symmetry_element_i<N, T>::get_type()
	 **/
	virtual const char *get_type() const {
		return k_sym_type;
	}

	/**	\copydoc symmetry_element_i<N, T>::clone()
	 **/
	virtual symmetry_element_i<N, T> *clone() const {
		return new se_label<N, T>(*this);
	}

	/**	\copydoc symmetry_element_i<N, T>::get_mask
	 **/
	virtual const mask<N> &get_mask() const {
		throw 1;
	}

	/**	\copydoc symmetry_element_i<N, T>::permute
	 **/
	virtual void permute(const permutation<N> &perm) {
		m_bidims.permute(perm);
		m_type.permute(perm);
		m_labels.permute(perm);
	}

	/**	\copydoc symmetry_element_i<N, T>::is_valid_bis
	 **/
	virtual bool is_valid_bis(const block_index_space<N> &bis) const;

	/**	\copydoc symmetry_element_i<N, T>::is_allowed
	 **/
	virtual bool is_allowed(const index<N> &idx) const;

	/**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
	 **/
	virtual void apply(index<N> &idx) const { }

	/**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
	 **/
	virtual void apply(index<N> &idx, transf<N, T> &tr) const { }

	//@}

};

template<size_t N, typename T>
const char se_label<N, T>::k_clazz = "se_label<N, T>";

template<size_t N, typename T>
const char se_label<N, T>::k_sym_type = "se_label";


template<size_t N, typename T>
se_label<N, T>::se_label(const block_index_space<N> &bis,
		const char *table_id, const label_t &target) :
	m_bidims(bis.get_block_index_dims()), m_label(target), m_labels(0)
	m_pt(product_table_container::get_instance().req_const_table(table_id))
{
	for (size_t i = 0; i < N; i++) {
		m_type[i] = bis.get_type(i);
		if (m_labels[i] == 0) {
			m_labels[i] = new label_list(m_bidims[i], m_pt.invalid());

			for (size_t j = i + 1; j < N; j++) {
				if (m_type[i] == bis.get_type(j)) m_labels[j] = m_labels[i];
			}
		}
	}
}

template<size_t N, typename T>
se_label<N, T>::se_label(const se_label<N, T> &elem) :
	m_bidims(elem.m_bidims), m_label(elem.m_label), m_pt(elem.m_pt),
	m_type(elem.m_type) {

	for (size_t i = 0; i < N; i++) {
		if (m_labels[i] == 0) {
			m_labels[i] = new label_list(m_bidims[i], m_pt.invalid());
			for (size_t j = 0; j < m_bidims[i]; j++)
				m_labels[i]->at(j) = elem.m_labels[i]->at(j);

			for (size_t j = i + 1; j < N; j++) {
				if (m_type[i] == bis.get_type(j)) m_labels[j] = m_labels[i];
			}
		}
	}

}

template<size_t N, typename T>
se_label<N, T>::~se_label() {

	for (size_t i = 0; i < N; i++) {
		delete m_labels[i];
		m_labels[i] = 0;

		for (size_t j = i + 1; j < N; j++) {
			if (m_type[i] == m_type[j]) {
				m_labels[j] = 0;
			}
		}
	}
}

template<size_t N, typename T>
void se_label<N, T>::assign(const mask<N> &msk, size_t pos,
		label_t label) throw(bad_parameter, out_of_bounds) {

	static const char *method = "assign(const mask<N> &, size_t, label_t)";

	if (! m_pt.is_valid(label)) {
		throw out_of_bounds(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Invalid label.");
	}

	size_t type;
	size_t i = 0;
	for (; i < N; i++) {
		if (msk[i]) break;
	}
	size_t j = i;
	type = m_type[i];
	for (; i < N; i++) {
		if (msk[i] && m_type[i] != type)
			throw bad_parameter(g_ns, k_clazz, method,
					__FILE__, __LINE__, "Invalid mask.");
	}

	if (pos >= m_bidims[i])
		throw out_of_bounds(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Position out of bounds.");

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);

	for (i = 0; i < N; i++) {
		if (i != j && m_type[i] == type) break;
	}
	if (i != N) {

		//introduce new type
	}


}

template<size_t N, typename T>
void se_label<N, T>::remove(const mask<N> &msk, size_t pos) throw(bad_parameter,
		out_of_bounds) {

	static const char *method = "remove(const mask<N> &, size_t)";

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}

template<size_t N, typename T>
void se_label<N, T>::clear() {

	for (size_t i = 0; i < N; i++) {
		size_t j = 0;
		for (; j < i; j++) {
			if (m_type[i] == m_type[j]) break;
		}
		if (i != j) continue;

		for (size_t l = 0; l < m_bidims[i]; l++)
			m_labels[i]->at(j) = m_pt.invalid();
	}
}

template<size_t N, typename T>
size_t se_label<N, T>::get_type(size_t dim) const  {

#ifdef LIBTENSOR_DEBUG
	if (dim > N)
		throw out_of_bounds(g_ns, k_clazz, "get_type(size_t)",
				__FILE__, __LINE__, "Dimension exceeds N.");
#endif

	return m_type[dim];
}

template<size_t N, typename T>
size_t se_label<N, T>::get_dim(size_t type) const throw(out_of_bounds) {

	size_t i = 0;
	for (; i < N; i++)
		if (type == m_type[i]) break;

	if (i == N)
		throw out_of_bounds(g_ns, k_clazz, "get_dim(size_t)",
				__FILE__, __LINE__, "Invalid type.");

	return m_bidims[i];

}

template<size_t N, typename T>
label_t se_label<N, T>::get_label(
		size_t type, size_t pos) const throw(out_of_bounds) {

	size_t i = 0;
	for (; i < N; i++)
		if (type == m_type[i]) break;

	if (i == N)
		throw out_of_bounds(g_ns, k_clazz, "get_dim(size_t)",
				__FILE__, __LINE__, "Invalid type.");

	if (pos >= m_bidims[i])
		throw out_of_bounds(g_ns, k_clazz, "get_dim(size_t)",
				__FILE__, __LINE__, "Position exceeds dimensions.");

	return m_labels[i]->at(pos);
}

template<size_t N, typename T>
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

	if (m_bidims != bis.get_block_index_dims()) return false;

	for (size_t i = 0; i < N; i++) {
		for (size_t j = i + 1; j < N; j++) {
			if (m_type(i) == m_type(j) && bis.get_type(i) != bis.get_type(j))
				return false;
		}
	}

	return true;
}

template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

	static const char *method = "is_allowed(const index<N> &)";

	label_group lg(4, m_pt.invalid());
	for (size_t i = 0; i < N; i++) {
		if (idx[i] >= m_bidims[i])
			throw out_of_bounds(g_ns, k_clazz, method,
					__FILE__, __LINE__, "Index out of bounds.");

		lg[i] = m_labels[i]->at(idx[i]);
	}

	return m_pt.is_in_product(lg, m_label);
}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_H

