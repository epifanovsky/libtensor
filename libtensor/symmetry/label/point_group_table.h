#ifndef LIBTENSOR_POINT_GROUP_TABLE_H
#define LIBTENSOR_POINT_GROUP_TABLE_H

#include <map>
#include "../../core/out_of_bounds.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Product tables for products of irreducible representations of a
		point group

	This implementation of product_table_i provides the product table for the
	irreducible representations of an arbitrary point symmetry group. The
	labels represent the different irreducible representations.

	The product table can be set up using the functions add_product() and
	delete_product(). Each product of two labels can have as many result labels
	as there are irreducible representations.

	The function \c is_in_product() computes the product of the label sequence
	given as first argument and compares the result to the label given as
	second argument. If the label is present in the result, true is returned.
	The product is evaluated from the right, i.e.
	\code
	l1 x (l2 x (l3 x ...))
	\endcode

    \ingroup libtensor_symmetry
 **/
class point_group_table : public product_table_i {
public:
    static const char *k_clazz; //!< Class name.

    typedef product_table_i::label_t irrep_label_t;
    typedef product_table_i::label_set_t irrep_set_t;
    typedef product_table_i::label_group_t irrep_group_t;
    typedef std::map<irrep_label_t, std::string> irrep_map_t;

private:
    typedef std::map<irrep_label_t, irrep_set_t> table_t;

    const std::string m_id; //!< Table id
    irrep_map_t m_irrep_names; //!< Human readable names of all irreps
    irrep_set_t m_irreps; //!< Complete set of all irreps
    irrep_label_t m_id_irrep; //!< Identity (totally symmetric) irrep
    table_t m_table; //!< The product table

public:
    //! \name Construction and destruction
    //@{

    /** \brief Constructor
        \param id Table ID
		\param irrep_names Names of all irreps
		\param ident Identity irrep (default: 0)
     **/
    point_group_table(const std::string &id,
            const irrep_map_t &irrep_names, irrep_label_t ident = 0);

    /** \brief Copy constructor
        \param pt Other point group table
     **/
    point_group_table(const point_group_table &pt);

    /** \brief Destructor
     **/
    virtual ~point_group_table() { }

    //@}

    //!	\name Implementation of product_table_i
    //@{

    /** \copydoc product_table_i::clone
     **/
    virtual point_group_table *clone() const {
        return new point_group_table(*this);
    }

    /** \copydoc product_table_i::get_id
     **/
    virtual const std::string &get_id() const {
        return m_id;
    }

    /** \copydoc product_table_i::is_valid
     **/
    virtual bool is_valid(irrep_label_t l) const {
        return m_irreps.count(l) != 0;
    }

    /** \copydoc product_table_i::get_complete_set
     **/
    virtual irrep_set_t get_complete_set() const {
        return m_irreps;
    }

    /** \copydoc product_table_i::get_identity
     **/
    virtual irrep_label_t get_identity() const {
        return m_id_irrep;
    }

    /** \copydoc product_table_i::is_in_product
     **/
    virtual bool is_in_product(const irrep_group_t &lg, irrep_label_t l) const;

    /** \copydoc product_table_i::product
     **/
    virtual irrep_set_t product(irrep_label_t l1, irrep_label_t l2) const {

        static const char *method = "product(irrep_label_t, irrep_label_t)";
#ifdef LIBTENSOR_DEBUG
        if (! is_valid(l1) && ! is_valid(l2))
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Invalid irrep.");
#endif

        table_t::const_iterator it = m_table.find(pair_index(l1, l2));
#ifdef LIBTENSOR_DEBUG
        if (it == m_table.end()) {
            throw generic_exception(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incomplete table.");
        }
#endif
        return it->second;
    }

    /** \copydoc product_table_i::product
     **/
    virtual irrep_set_t product(
            irrep_label_t l1, const irrep_set_t &ls2) const {

        irrep_set_t ls1; ls1.insert(l1);
        return product(ls1, ls2);
    }

    /** \copydoc product_table_i::product
     **/
    virtual irrep_set_t product(
            const irrep_set_t &ls1, irrep_label_t l2) const {

        irrep_set_t ls2; ls2.insert(l2);
        return product(ls1, ls2);
    }

    /** \copydoc product_table_i::product
     **/
    virtual irrep_set_t product(
            const irrep_set_t &l1, const irrep_set_t &l2) const;

    /** \copydoc product_table_i::is_in_product
     **/
    virtual void check() const throw(generic_exception);

    //@}

    //!	\name Manipulation functions
    //@{

    /** \brief Adds product of two irreps.
		\param l1 First label
		\param l2 Second label
		\param lr Result label
		\throw out_of_bounds If l1, l2 or lr are not valid.
     **/
    void add_product(irrep_label_t l1, irrep_label_t l2,
            irrep_label_t lr) throw(bad_parameter);

    /** \brief Reset the product table
     **/
    void reset() {
        m_table.clear();
        init_table();
    }

    //@}

    const std::string &irrep_name(irrep_label_t l1) const throw(bad_parameter) {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid(l1))
            throw bad_parameter(g_ns, k_clazz, "irrep_name(irrep_label_t)",
                    __FILE__, __LINE__, "Invalid irrep.");
#endif
        return m_irrep_names.find(l1)->second;
    }

private:
    irrep_label_t pair_index(irrep_label_t l1, irrep_label_t l2) const {
        if (l1 > l2)
            return l1 * (l1 + 1) / 2 + l2;
        else
            return l2 * (l2 + 1) / 2 + l1;
    }

    void init_table();
};

}

#endif // LIBTENSOR_POINT_GROUP_TABLE_H
