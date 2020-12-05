#pragma once
#include <string>
#include <vector>

/** \brief Version of the %tensor library

    The %version of the library is specified using the %version number and
    status. The number consists of a major number and a minor number. The
    status string describes the release status.
*/
namespace libtensor {

/**Version information */
struct metadata {
  /** Return the major part of the version */
  static int major_part();

  /** Return the minor part of the version */
  static int minor_part();

  /** Return the patch part of the version */
  static int patch_part();

  /** Is the compiled version a Debug version */
  static bool is_debug();

  /** The compiled-in optional features */
  static std::vector<std::string> features();

  /** The compiled-in BLAS backend */
  static std::string blas();

  /** The package authors */
  static std::string authors();

  /**  Return the version as a string */
  static std::string version_string();
};

///@}
}  // namespace libtensor
