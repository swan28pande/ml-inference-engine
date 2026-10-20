# Locates Protobuf + gRPC and provides mie_generate_grpc() for compiling .proto
# files. Prefers CMake config packages (vcpkg, Homebrew, source builds) and
# falls back to pkg-config for distro packages that do not ship them.

find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf REQUIRED)
endif()

find_package(gRPC CONFIG QUIET)
if(gRPC_FOUND)
  set(MIE_GRPCPP_TARGET gRPC::grpc++)
  if(TARGET gRPC::grpc_cpp_plugin)
    set(MIE_GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  set(MIE_GRPCPP_TARGET PkgConfig::GRPCPP)
endif()

if(NOT MIE_GRPC_CPP_PLUGIN)
  find_program(MIE_GRPC_CPP_PLUGIN_PATH grpc_cpp_plugin REQUIRED)
  set(MIE_GRPC_CPP_PLUGIN ${MIE_GRPC_CPP_PLUGIN_PATH})
endif()

if(TARGET protobuf::protoc)
  set(MIE_PROTOC $<TARGET_FILE:protobuf::protoc>)
else()
  set(MIE_PROTOC ${Protobuf_PROTOC_EXECUTABLE})
endif()

if(TARGET protobuf::libprotobuf)
  set(MIE_PROTOBUF_TARGET protobuf::libprotobuf)
else()
  set(MIE_PROTOBUF_TARGET ${Protobuf_LIBRARIES})
endif()

# mie_generate_grpc(<out_srcs_var> <proto_dir> <proto files...>)
function(mie_generate_grpc out_var proto_dir)
  set(gen_dir ${CMAKE_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${gen_dir})
  set(outputs)
  foreach(proto ${ARGN})
    get_filename_component(abs ${proto} ABSOLUTE)
    get_filename_component(stem ${proto} NAME_WE)
    set(pb_src  ${gen_dir}/${stem}.pb.cc)
    set(pb_hdr  ${gen_dir}/${stem}.pb.h)
    set(rpc_src ${gen_dir}/${stem}.grpc.pb.cc)
    set(rpc_hdr ${gen_dir}/${stem}.grpc.pb.h)
    add_custom_command(
      OUTPUT ${pb_src} ${pb_hdr} ${rpc_src} ${rpc_hdr}
      COMMAND ${MIE_PROTOC}
        --proto_path=${proto_dir}
        --cpp_out=${gen_dir}
        --grpc_out=${gen_dir}
        --plugin=protoc-gen-grpc=${MIE_GRPC_CPP_PLUGIN}
        ${abs}
      DEPENDS ${abs}
      COMMENT "Generating C++/gRPC sources for ${stem}.proto"
      VERBATIM)
    list(APPEND outputs ${pb_src} ${rpc_src})
  endforeach()
  set(${out_var} ${outputs} PARENT_SCOPE)
endfunction()
