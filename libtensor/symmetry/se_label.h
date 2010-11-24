#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include "../defs.h"
#include "../core/block_index_space.h"
#include "../core/mask.h"
#include "../core/symmetry_element_i.h"
#include "../not_implemented.h"
#include "product_table_container.h"

namespace libtensor {

/**	\brief Symmetry element for label assigned to %tensor blocks
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	This %symmetry elements establishes a labeling of blocks along each
	dimension of a block %tensor and a mapping of N labels onto labels of
	%tensor blocks via a product table.

	The block labels have to be setup via the functions assign, remove
	and clear (see description there). Blocks and / or dimensions can also be
	left unassigned. From the labels along the dimensions labels for %tensor
	blocks are obtained by using the product table. By comparison of the
	resulting labels to the target label %tensor blocks are accepted (if labels
	match) or discarded. Unassigned labels are treated as if they represent any
	label, i.e. the respective %tensor blocks are always accepted.

	The target label can comprise any number of labels. If no target label is
	set, all blocks are discarded. If the target label comprises all valid
	labels, all blocks are accepted.

	The product table (given by the table_id in the constructor) is obtained
	from the product table container object at time of construction of an
	se_label object and returned after destruction. Thus, a product table has
	to be setup before a se_label object using it is created and cannot be
	altered as long as such a se_label object exists.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_label : public symmetry_element_i<N, T> {
public:
	static const char *k_clazz; //!< Class name
	static const char *k_sym_type; //!< Symmetry type

	typedef product_table_i::label_t label_t;
	typedef product_table_i::label_group label_group;

private:

	dimensions<N> m_bidims; //!< Block index dimensions
	sequence<N, size_t> m_type; //!< Label type
	sequence<N, label_group*> m_labels; //!< Block labels
	label_group m_target; //!< Target label

	const product_table_i &m_pt; //!< Product table
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the %symmetry element
		\param bidims Block %index dimensions.
		\param id Id of product table.
	 **/
	se_label(const dimensions<N> &bidims, const std::string &id);

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

	/** \brief Match labels
	 **/
	void match_labels();

	/** \brief Adds a target label.
		\throw bad_parameter If target is invalid.
	 **/
	void add_target(label_t target) throw(bad_parameter);

	/** \brief Deletes all target labels set previously.
	 **/
	void delete_target();

	//@}

	//! \name Data access
	//@{

	/**	\brief Returns the block index dimensions of assigned to the se_label
	 **/
	const dimensions<N> &get_block_index_dims() const {
		return m_bidims;
	}

	/**	\brief Returns the type (labeling pattern) of a dimension
		\param dim Dimension number.
		\throw out_of_bounds If the dimension number is out of bounds.
	 **/
	size_t get_dim_type(size_t dim) const;

	/**	\brief Returns the block dimension of a dimension type
		\param type Dimension type.
		\throw bad_parameter If the dimension type is invalid.
	 **/
	size_t get_dim(size_t type) const throw(bad_parameter);

	/** \brief Returns the number of valid labels.
	 **/
	size_t get_n_labels() const {
		return m_pt.nlabels();
	}

	/** \brief Checks whether a label is valid.
	 **/
	bool is_valid(label_t l) const {
		return m_pt.is_valid(l);
	}


	/**	\brief Returns the label of a block of a dimension type.
		\param type Dimension type.
		\param pos Block position.
		\throw out_of_bounds If the dimension type is out of bounds.
	 **/
	label_t get_label(size_t type, size_t pos) const throw(out_of_bounds);

	/** \brief Returns the number of target labels
	 **/
	size_t get_n_targets() const {
		return m_target.size();
	}

	/** \brief Returns the i-th target label
	 	\param i Number of target label (default i = 0)
	 	\throw out_of_bounds If i is out of bounds
	 **/
	label_t get_target(size_t i = 0) const throw(out_of_bounds) {
#ifdef LIBTENSOR_DEBUG
		if (i >= m_target.size())
			throw out_of_bounds(g_ns, k_clazz, "get_target(size_t)",
					__FILE__, __LINE__, "Unknown target label.");
#endif

		return m_target[i];
	}

	const std::string &get_table_id() const {
		return m_pt.get_id();
	}
	//@}

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

	/**	\copydoc symmetry_element_i<N, T>::permute
	 **/
	void permute(const permutation<N> &perm) {
		m_bidims.permute(perm);
		m_type.permute(perm);
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
const char *se_label<N, T>::k_clazz = "se_label<N, T>";

template<size_t N, typename T>
const char *se_label<N, T>::k_sym_type = "se_label";


template<size_t N, typename T>
se_label<N, T>::se_label(const dimensions<N> &bidims, const std::string &id) :
	m_bidims(bidims), m_type(0), m_labels(0),
	m_pt(product_table_container::get_instance().req_const_table(id)) {

	mask<N> done;
	size_t curr_type = 0;
	for (size_t i = 0; i < N; i++) {
		if (done[i]) continue;

		done[i] = true;
		m_type[i] = curr_type;
		m_labels[curr_type] = new label_group(m_bidims[i], m_pt.invalid());

		for (size_t j = i + 1; j < N; j++) {
			if (m_bidims[i] == m_bidims[j]) {
				m_type[j] = curr_type;
				done[j] = true;
			}
		}
		curr_type++;
	}
}

template<size_t N, typename T>
se_label<N, T>::se_label(const se_label<N, T> &el) :
	m_bidims(el.m_bidims), m_type(el.m_type), m_labels(0),
	m_pt(product_table_container::get_instance().req_const_table(
			el.m_pt.get_id())), m_target(el.m_target) {

	for (size_t itype = 0; itype < N; itype++) {
		if (el.m_labels[itype] == 0) break;

		m_labels[itype] = new label_group(*(el.m_labels[itype]));
	}
}

template<size_t N, typename T>
se_label<N, T>::~se_label() {

	for (size_t i = 0; i < N; i++) {
		if (m_labels[i] == 0) break;

		delete m_labels[i];
		m_labels[i] = 0;
	}
	product_table_container::get_instance().ret_table(m_pt.get_id());
}

template<size_t N, typename T>
void se_label<N, T>::assign(const mask<N> &msk, size_t pos,
		label_t label) throw(bad_parameter, out_of_bounds) {

	static const char *method = "assign(const mask<N> &, size_t, label_t)";

	if (! m_pt.is_valid(label)) {
		throw out_of_bounds(g_ns, k_clazz,
				method, __FILE__, __LINE__, "Invalid label.");
	}

	size_t i = 0, type;
	for (; i < N; i++)  if (msk[i]) break;
	if (i == N) return; // mask has no true component
	type = m_type[i];
	if (pos >= m_labels[type]->size()) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__,
				__LINE__, "Labeling position is out of bounds.");
	}

	bool adjust = false;
	for(i = 0; i < N; i++) {
		if(msk[i]) {
			if(m_type[i] != type) break;
		} else {
			if(m_type[i] == type) adjust = true;
		}
	}
	if(i != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Invalid labeling mask.");
	}

	label_group *labels = 0;
	if (adjust) {
		size_t new_type = 0;
		for (i = 0; i < N; i++)
			new_type = std::max(new_type, m_type[i]);
		new_type++;

		m_labels[new_type] = labels = new label_group(*(m_labels[type]));
		for (i = 0; i < N; i++)
			if (msk[i]) m_type[i] = new_type;
	}
	else {
		labels = m_labels[type];
	}

	labels->at(pos) = label;

}

template<size_t N, typename T>
void se_label<N, T>::remove(const mask<N> &msk, size_t pos) throw(bad_parameter,
		out_of_bounds) {

	static const char *method = "remove(const mask<N> &, size_t)";

	size_t i = 0, type;
	for (; i < N; i++)  if (msk[i]) break;
	if (i == N) return; // mask has no true component
	type = m_type[i];

	if (pos >= m_labels[type]->size()) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__,
				__LINE__, "Position is out of bounds.");
	}

	bool new_type = false;
	for(i = 0; i < N; i++) {
		if(msk[i]) {
			if(m_type[i] != type) break;
		} else {
			if(m_type[i] == type) new_type = true;
		}
	}
	if(i != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Invalid labeling mask.");
	}

	label_group *labels = 0;
	if (new_type) {
		size_t max_type = 0;
		for (i = 0; i < N; i++)
			max_type = std::max(max_type, m_type[i]);
		max_type++;

		m_labels[max_type] = labels = new label_group(*(m_labels[type]));
		for (i = 0; i < N; i++)
			if (msk[i]) m_type[i] = max_type;
	}
	else {
		labels = m_labels[type];
	}
	labels->at(pos) = m_pt.invalid();
}

template<size_t N, typename T>
void se_label<N, T>::clear() {

	for (size_t i = 0; i < N; i++) {
		if (m_labels[i] == 0) break;

		for (size_t j = 0; j < m_labels[i]->size(); j++)
			m_labels[i]->at(j) = m_pt.invalid();
	}
}

template<size_t N, typename T>
void se_label<N, T>::match_labels() {

	sequence<N, size_t> types(m_type);
	sequence<N, label_group*> labels(m_labels);

	for (size_t i = 0; i < N; i++) {
		m_type[i] = 0;
		m_labels[i] = 0;
	}

	size_t curr_type = 0;
	for (size_t i = 0; i < N; i++) {
		size_t itype = types[i];
		if (labels[itype] == 0) continue;

		m_type[i] = curr_type;
		label_group *lli = m_labels[curr_type] = labels[itype];
		labels[itype] = 0;

		for (size_t j = i + 1; j < N; j++) {
			size_t jtype = types[j];
			if (itype == jtype) {
				m_type[j] = curr_type;
				continue;
			}

			if (labels[jtype] == 0) continue;
			if (lli->size() != labels[jtype]->size()) continue;

			size_t k = 0;
			for (; k < lli->size(); k++) {
				if (lli->at(k) != labels[jtype]->at(k)) break;
			}
			if (k != lli->size()) continue;

			delete labels[jtype];
			labels[jtype] = 0;
			m_type[j] = curr_type;
		}

		curr_type++;
	}
}

template<size_t N, typename T>
void se_label<N, T>::add_target(label_t target) throw(bad_parameter) {

	if (! m_pt.is_valid(target))
		throw bad_parameter(g_ns, k_clazz,
				"set_label(label_t)", __FILE__, __LINE__, "Invalid label.");

	label_group::iterator i = m_target.begin();
	while (i != m_target.end()) {
		if (*i == target) return;
		if (*i > target) break;

		i++;
	}

	m_target.insert(i, target);
}

template<size_t N, typename T>
void se_label<N, T>::delete_target() {

	m_target.clear();
}


template<size_t N, typename T>
size_t se_label<N, T>::get_dim_type(size_t dim) const  {

#ifdef LIBTENSOR_DEBUG
	if (dim > N)
		throw out_of_bounds(g_ns, k_clazz, "get_type(size_t)",
				__FILE__, __LINE__, "Dimension exceeds N.");
#endif

	return m_type[dim];
}

template<size_t N, typename T>
size_t se_label<N, T>::get_dim(size_t type) const throw(bad_parameter) {

	if (type > N || m_labels[type] == 0)
		throw out_of_bounds(g_ns, k_clazz, "get_dim(size_t)",
				__FILE__, __LINE__, "Invalid type.");

	return m_labels[type]->size();

}

template<size_t N, typename T>
typename se_label<N, T>::label_t se_label<N, T>::get_label(
		size_t type, size_t pos) const throw(out_of_bounds) {

	static const char *method = "get_label(size_t, size_t)";

	if (type > N || m_labels[type] == 0)
		throw out_of_bounds(g_ns, k_clazz,
				method, __FILE__, __LINE__, "Invalid type.");

	if (pos >= m_labels[type]->size())
		throw out_of_bounds(g_ns, k_clazz,
				method, __FILE__, __LINE__, "Position exceeds dimensions.");

	return m_labels[type]->at(pos);
}

template<size_t N, typename T>
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

	const dimensions<N> &bidims = bis.get_block_index_dims();

	for (size_t i = 0; i < N; i++)
		if (m_labels[m_type[i]]->size() != bidims[i])
			return false;

	return true;
}

template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

	static const char *method = "is_allowed(const index<N> &)";

	if (m_target.size() == 0) return false;
	if (m_target.size() == m_pt.nlabels()) return true;

	label_group lg(N, m_pt.invalid());
	for (size_t i = 0; i < N; i++) {
		if (idx[i] >= m_labels[m_type[i]]->size())
			throw out_of_bounds(g_ns, k_clazz, method,
					__FILE__, __LINE__, "Index out of bounds.");

		lg[i] = m_labels[m_type[i]]->at(idx[i]);
		if (! m_pt.is_valid(lg[i]))
			return true;
	}

	for (label_group::const_iterator i = m_target.begin();
			i != m_target.end(); i++) {
		if (m_pt.is_in_product(lg, *i))
			return true;
	}

	return false;
}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_H

