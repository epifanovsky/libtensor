#ifndef LIBTENSOR_BAD_BLOCK_INDEX_SPACE_H
#define LIBTENSOR_BAD_BLOCK_INDEX_SPACE_H

#include "../exception.h"

namespace libtensor {


/** \brief Exception indicating that a block %tensor passed to a block
        %tensor operation has incorrect block %index space

    \ingroup libtensor_btod
 **/
class bad_block_index_space : public exception_base<bad_block_index_space> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    bad_block_index_space(const char *ns, const char *clazz,
        const char *method, const char *file, unsigned int line,
        const char *message) throw() :
        exception_base<bad_block_index_space>(ns, clazz, method,
            file, line, "bad_block_index_space", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~bad_block_index_space() throw() { };

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_BAD_BLOCK_INDEX_SPACE_H
