#ifndef LIBTENSOR_LINALG_H
#define LIBTENSOR_LINALG_H

#include "linalg_cblas_level1.h"
#include "linalg_cblas_level2.h"
#include "linalg_cblas_level3.h"

namespace libtensor {

/** \brief Linear algebra implementation based on CBLAS

    \ingroup libtensor_linalg
 **/
class linalg : public linalg_cblas_level1,
               public linalg_cblas_level2,
               public linalg_cblas_level3 {

 public:
  typedef double element_type;        //!< Data type
  typedef void* device_context_type;  //!< Device context
  typedef void* device_context_ref;   //!< Reference type to device context
};

}  // namespace libtensor

#endif  // LIBTENSOR_LINALG_H
