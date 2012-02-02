#ifndef LIBTENSOR_LETTER_H
#define LIBTENSOR_LETTER_H

namespace libtensor {


/** \brief Identifies a letter tensor index

    This is an empty class that serves the purpose of identifying
    the letter indexes of tensors in tensor expressions.

    Letter indexes can be combined using the bitwise or (|) operators.

    \ingroup libtensor_expr
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


} // namespace libtensor

#endif // LIBTENSOR_LETTER_H

