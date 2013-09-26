#ifndef LIBTENSOR_IFACE_LABELED_H
#define LIBTENSOR_IFACE_LABELED_H

#include <cstddef> // for size_t
#include "letter_expr.h"

namespace libtensor {
namespace iface {


/** \brief Someting that has a letter label attached to it

    \ingroup libtensor_iface
 **/
template<size_t N>
class labeled {
public:
    const letter_expr<N> &get_label() const;

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_LABELED_H
