# Compiler warnings and sanitizer configuration shared by all project targets.

add_library(mie_compile_options INTERFACE)
add_library(mie::compile_options ALIAS mie_compile_options)

if(MSVC)
  target_compile_options(mie_compile_options INTERFACE /W4 /permissive-)
  if(MIE_WARNINGS_AS_ERRORS)
    target_compile_options(mie_compile_options INTERFACE /WX)
  endif()
else()
  target_compile_options(mie_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor>)
  if(MIE_WARNINGS_AS_ERRORS)
    target_compile_options(mie_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

if(MIE_SANITIZER AND NOT MSVC)
  if(MIE_SANITIZER STREQUAL "address")
    set(_mie_san_flags -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  elseif(MIE_SANITIZER STREQUAL "thread")
    set(_mie_san_flags -fsanitize=thread -fno-omit-frame-pointer)
  else()
    message(FATAL_ERROR "MIE_SANITIZER must be 'address' or 'thread' (got '${MIE_SANITIZER}')")
  endif()
  # Applied globally so third-party code built from source (GoogleTest) is instrumented too.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${_mie_san_flags}>")
  add_link_options(${_mie_san_flags})
endif()
