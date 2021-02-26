#ifndef LIBTENSOR_POINT_GROUP_TABLE_H
#define LIBTENSOR_POINT_GROUP_TABLE_H

#include <vector>
#include <libtensor/core/out_of_bounds.h>
#include "product_table_i.h"

namespace libtensor {

/** \brief Product tables for products of irreducible representations of
        a point group

    This implementation of product_table_i provides the product table for
    the irreducible representations (irreps) of an arbitrary point group.
    Upon construction the complete list of irreps is passed to the object
    together with a unique ID of the table (usually the name of the point
    group) and the name of the identity (totally symmetric) irrep. Based
    on this information the constructor then assigns labels to the irreps
    so that the totally symmetric irrep has label 0. This assignment can
    latter be accessed by the functions \c get_irrep_name() and
    \c get_label(). Furthermore, the constructor initialized the product
    table such that it fulfills the requirement for the identity label
    imposed by product_table_i.

    To setup the product table the functions \c add_product() and \c reset()
    are provided.

    \ingroup libtensor_symmetry
 **/
class point_group_table : public product_table_i {
public:
    static const char *k_clazz; //!< Class name.

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;

private:
    const std::string m_id; //!< Table id
    std::vector<std::string> m_irreps; //!< Maximum number of labels
    std::vector<size_t> m_table; //!< The product table

public:
    //! \name Constructors / destructor
    //@{

    /** \brief Constructor
        \param id Table ID
        \param irreps Names of all irreps
        \param ident Name of the identity irrep
     **/
    point_group_table(const std::string &id,
            const std::vector<std::string> &irreps,
            const std::string &identity);

    /** \brief Destructor
     **/
    virtual ~point_group_table() { }

    //@}

    //!    \name Implementation of product_table_i
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
    virtual bool is_valid(label_t l) const {
        return l < m_irreps.size();
    }


    /** \copydoc product_table_i::get_n_labels
     **/
    virtual label_t get_n_labels() const {
        return m_irreps.size();
    }


    /** \copydoc product_table_i::product
     **/
    virtual void product(const label_group_t &lg, label_set_t &prod) const;


    /** \copydoc product_table_i::is_in_product
     **/
    virtual bool is_in_product(const label_group_t &lg, label_t l) const;

    //@}

    /** \brief Return the irrep name for the given label
     **/
    const std::string &get_irrep_name(label_t l) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid(l))
            throw bad_parameter(g_ns, k_clazz, "get_irrep_name(label_t) const",
                    __FILE__, __LINE__, "Invalid label.");
#endif

        return m_irreps[l];
    }

    /** \brief Return the label for a given irrep name
     **/
    label_t get_label(const std::string &irrep) const;


    //!    \name Manipulation functions
    //@{

    /** \brief Adds product of two irreps.
        \param l1 First label
        \param l2 Second label
        \param lr Result label
        \throw out_of_bounds If l1, l2 or lr are not valid.
     **/
    void add_product(label_t l1, label_t l2, label_t lr);

    /** \brief Reset the product table
     **/
    void reset() {
        m_table.clear();
        m_table.resize(m_irreps.size() * (m_irreps.size() + 1) / 2);
        initialize_table();
    }

    //@}

protected:
    virtual void do_check() const { }

private:
    static label_t pair_index(label_t l1, label_t l2) {
        return l2 * (l2 + 1) / 2 + l1;
    }

    void initialize_table();
};

}

#endif // LIBTENSOR_POINT_GROUP_TABLE_H
