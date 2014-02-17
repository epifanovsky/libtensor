#ifndef LIBTENSOR_EXPR_LETTER_H
#define LIBTENSOR_EXPR_LETTER_H

namespace libtensor {
namespace expr {


/** \brief Identifies a letter %tensor %index

    This is an empty class that serves the purpose of identifying
    %letter indexes of a %tensor in %tensor expressions.

    Letter indexes can be combined using the multiplication (*) and the
    bitwise or (|) operators.

    \sa label

    \ingroup libtensor_expr_iface
**/
class letter {
public:
    bool operator==(const letter &other) const {
        return this == &other;
    }

    bool operator!=(const letter &other) const {
        return this != &other;
    }
};


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::letter;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_LETTER_H
