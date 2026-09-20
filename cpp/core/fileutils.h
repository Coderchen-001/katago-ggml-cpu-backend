#ifndef CORE_FILEUTILS_H_
#define CORE_FILEUTILS_H_

#include <cstdio>

#include "../core/global.h"

namespace FileUtils {
  bool exists(const std::string& path);

  // Open a C FILE* on a UTF-8 path through the SAME conversion every other function here uses
  // (ghc::filesystem's u8path).  Exists because the tuner's atomic writer needs a FILE* -- for the
  // fflush + fsync durability an ofstream cannot give -- while still agreeing with
  // tryRename/tryRemoveFile about what the path names.  Opening with the ANSI fopen while renaming
  // through this module was the r6 defect: on a non-ASCII path the write succeeded and the rename
  // could not see the file.  Returns nullptr on failure; the caller fclose()s it.
  std::FILE* tryFopen(const std::string& filename, const char* mode);

  // Read a PATH-VALUED environment variable as UTF-8, and the ONE place that does so (r6).  On
  // Windows MSVC's getenv narrows the UTF-16 process environment to the ANSI code page, so a
  // non-ASCII path arrives as ANSI bytes that every consumer here would then read as UTF-8: one
  // string, two different files, depending on which layer looks at it.  Empty when unset or empty.
  std::string readPathEnv(const char* name);

  // Returns whether good() is true on the fstream after attempting to call open on it.
  bool tryOpen(std::ifstream& in, const char* filename, std::ios_base::openmode mode = std::ios_base::in);
  bool tryOpen(std::ofstream& out, const char* filename, std::ios_base::openmode mode = std::ios_base::out);
  bool tryOpen(std::ifstream& in, const std::string& filename, std::ios_base::openmode mode = std::ios_base::in);
  bool tryOpen(std::ofstream& out, const std::string& filename, std::ios_base::openmode mode = std::ios_base::out);
  // Raises exception unless file open was successful
  void open(std::ifstream& in, const char* filename, std::ios_base::openmode mode = std::ios_base::in);
  void open(std::ofstream& out, const char* filename, std::ios_base::openmode mode = std::ios_base::out);
  void open(std::ifstream& in, const std::string& filename, std::ios_base::openmode mode = std::ios_base::in);
  void open(std::ofstream& out, const std::string& filename, std::ios_base::openmode mode = std::ios_base::out);

  void loadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf);
  void loadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf, std::string* actualSha256Buf);
  void uncompressAndLoadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf);
  void uncompressAndLoadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf, std::string* actualSha256Buf);

  bool isDirectory(const std::string& filename);
  bool tryRemoveFile(const std::string& filename);

  std::string weaklyCanonical(const std::string& path);

  bool tryRename(const std::string& src, const std::string& dst);
  void rename(const std::string& src, const std::string& dst);

  // Read entire file whole
  std::string readFile(const char* filename);
  std::string readFile(const std::string& filename);
  std::string readFileBinary(const char* filename);
  std::string readFileBinary(const std::string& filename);

  // Read file into separate lines, using the specified delimiter character(s).
  // The delimiter characters are NOT included.
  std::vector<std::string> readFileLines(const char* filename, char delimiter);
  std::vector<std::string> readFileLines(const std::string& filename, char delimiter);

  // Return list of filenames of files in a directory
  std::vector<std::string> listFiles(const std::string& dirname);

  // Recursively walk a directory and find all the files that match fileFilter.
  // fileFilter receives just the file name and not the full path, but collected contains the paths.
  void collectFiles(const std::string& dirname, const std::function<bool(const std::string&)>& fileFilter, std::vector<std::string>& collected);
}


#endif // CORE_FILEUTILS_H_
