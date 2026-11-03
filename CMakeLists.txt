cmake_minimum_required(VERSION 3.20)

project(mini_gpt VERSION 0.1 LANGUAGES CXX)

add_library(mini_gpt_tensor STATIC
    src/tensor.cpp
)

target_include_directories(mini_gpt_tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(mini_gpt_tensor PUBLIC cxx_std_20)
set_target_properties(mini_gpt_tensor PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_tensor PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_tensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_tokenizer STATIC
    src/bpe_tokenizer.cpp
)

target_include_directories(mini_gpt_tokenizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(mini_gpt_tokenizer PUBLIC cxx_std_20)
set_target_properties(mini_gpt_tokenizer PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_tokenizer PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_embedding STATIC
    src/embedding.cpp
)

target_include_directories(mini_gpt_embedding
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_embedding PUBLIC mini_gpt_tensor)
target_compile_features(mini_gpt_embedding PUBLIC cxx_std_20)
set_target_properties(mini_gpt_embedding PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_embedding PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_embedding PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_attention STATIC
    src/causal_self_attention.cpp
)

target_include_directories(mini_gpt_attention
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_attention PUBLIC mini_gpt_tensor)
target_compile_features(mini_gpt_attention PUBLIC cxx_std_20)
set_target_properties(mini_gpt_attention PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_attention PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_attention PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(tokenizer_demo
    apps/tokenizer_demo.cpp
)

target_link_libraries(tokenizer_demo PRIVATE mini_gpt_tokenizer)
target_compile_features(tokenizer_demo PRIVATE cxx_std_20)
target_compile_definitions(tokenizer_demo PRIVATE
    MINI_GPT_DEMO_CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/conversations.txt"
)
set_target_properties(tokenizer_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(tokenizer_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(tokenizer_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(embedding_demo
    apps/embedding_demo.cpp
)

target_link_libraries(embedding_demo
    PRIVATE
        mini_gpt_embedding
        mini_gpt_tokenizer
)
target_compile_features(embedding_demo PRIVATE cxx_std_20)
set_target_properties(embedding_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(embedding_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(embedding_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(attention_demo
    apps/attention_demo.cpp
)

target_link_libraries(attention_demo
    PRIVATE
        mini_gpt_attention
        mini_gpt_embedding
        mini_gpt_tokenizer
)
target_compile_features(attention_demo PRIVATE cxx_std_20)
set_target_properties(attention_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(attention_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(attention_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(tensor_tests
        tests/tensor_tests.cpp
    )

    target_link_libraries(tensor_tests PRIVATE mini_gpt_tensor)
    set_target_properties(tensor_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(tensor_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(tensor_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME tensor_tests COMMAND tensor_tests)

    add_executable(tokenizer_tests
        tests/tokenizer_tests.cpp
    )

    target_link_libraries(tokenizer_tests PRIVATE mini_gpt_tokenizer)
    target_compile_definitions(tokenizer_tests PRIVATE
        MINI_GPT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )
    set_target_properties(tokenizer_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(tokenizer_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(tokenizer_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME tokenizer_tests COMMAND tokenizer_tests)

    add_executable(embedding_tests
        tests/embedding_tests.cpp
    )

    target_link_libraries(embedding_tests PRIVATE mini_gpt_embedding)
    set_target_properties(embedding_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(embedding_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(embedding_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME embedding_tests COMMAND embedding_tests)

    add_executable(attention_tests
        tests/attention_tests.cpp
    )

    target_link_libraries(attention_tests PRIVATE mini_gpt_attention)
    set_target_properties(attention_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(attention_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(attention_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME attention_tests COMMAND attention_tests)
endif()
