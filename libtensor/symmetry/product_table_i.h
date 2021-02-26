#ifndef LIBTENSOR_PRODUCT_TABLE_I_H
#define LIBTENSOR_PRODUCT_TABLE_I_H

#include <set>
#include <string>
#include <vector>
#include "bad_symmetry.h"

namespace libtensor {


/** \brief Interface for product tables

    This class defines the interface for a product table of a finite set of
    labels. The provided functions to access the product table enforce that
    the table is symmetric. Additionally, two special labels are defined as
    constants: the identity label (0) and an invalid label ((size_t) -1).
    The identity label is the label for which the product with any other
    label l yields only l as result.

    Any class implementing this interface needs to implement
    - clone() -- Create an identical copy
    - get_id() -- Return the unique ID of the current table
    - is_valid() -- Check if the given label is valid
    - get_n_labels() -- Return the total number of labels
    - determine_product() -- Compute the product of two labels
    - do_check() -- Perform additional checks

    \ingroup libtensor_symmetry
 **/
class product_table_i {
public:
    typedef size_t label_t; //!< Label type
    typedef std::set<label_t> label_set_t; //!< Set of unique labels
    typedef std::vector<label_t> label_group_t; //!< Group of labels

    static const char *k_clazz; //!< Class name
    static const label_t k_invalid; //!< Invalid label
    static const label_t k_identity; //!< Identity label

public:
    //! \name Constructors / destructors
    //@{

    /** \brief Virtual destructor
     **/
    virtual ~product_table_i() { }

    /** \brief Creates an identical copy of the product table.
     **/
    virtual product_table_i *clone() const = 0;

    //@}

    /** \brief Returns the id identifying the product table.
     **/
    virtual const std::string &get_id() const = 0;

    /** \brief Returns true if label is valid
        \param l Label to check
     **/
    virtual bool is_valid(label_t l) const = 0;

    /** \brief Returns the number of valid labels (which is the smallest label
            not valid).
     **/
    virtual label_t get_n_labels() const = 0;

    /** \brief Computes the product of a label group
        \param lg Label group.
        \param[out] prod Computed product.

        The result is the product of all n labels in the group
        \f$ l_1 \times l_2 \times ... \times l_n \f$
     **/
    virtual void product(const label_group_t &lg, label_set_t &prod) const = 0;

    /** \brief Determines if the label is in the product.
        \param lg Group of labels to take the product of.
        \param l Label to check against.
        \return True if label is in the product, else false.
     **/
    virtual bool is_in_product(const label_group_t &lg, label_t l) const = 0;

    /** \brief Does a consistency check on the table.
        \throw exception If product table is not set up properly.

        Checks that the product of any label with the identity label yields the
        respective label and that all products yield a non-empty set.
     **/
    void check() const;

protected:
    /** \brief Perform additional consistency check

        This function is called by \c check() to perform additional checks
        for the child classes.
     **/
    virtual void do_check() const = 0;
};

}

#endif // LIBTENSOR_PRODUCT_TABLE_I_H
