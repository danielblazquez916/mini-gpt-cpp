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

add_library(mini_gpt_layer_norm STATIC
    src/layer_norm.cpp
)

target_include_directories(mini_gpt_layer_norm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_layer_norm PUBLIC mini_gpt_tensor)
target_compile_features(mini_gpt_layer_norm PUBLIC cxx_std_20)
set_target_properties(mini_gpt_layer_norm PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_layer_norm PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_layer_norm PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_transformer_block STATIC
    src/transformer_block.cpp
)

target_include_directories(mini_gpt_transformer_block
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_transformer_block
    PUBLIC
        mini_gpt_attention
        mini_gpt_layer_norm
)
target_compile_features(mini_gpt_transformer_block PUBLIC cxx_std_20)
set_target_properties(mini_gpt_transformer_block PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_transformer_block PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_transformer_block PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_gpt STATIC
    src/gpt.cpp
)

target_include_directories(mini_gpt_gpt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_gpt
    PUBLIC
        mini_gpt_embedding
        mini_gpt_transformer_block
        mini_gpt_tokenizer
)
target_compile_features(mini_gpt_gpt PUBLIC cxx_std_20)
set_target_properties(mini_gpt_gpt PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_gpt PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_gpt PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_loss STATIC
    src/cross_entropy.cpp
)

target_include_directories(mini_gpt_loss
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_loss PUBLIC mini_gpt_tensor)
target_compile_features(mini_gpt_loss PUBLIC cxx_std_20)
set_target_properties(mini_gpt_loss PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_loss PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_loss PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_inference STATIC
    src/inference.cpp
)

target_include_directories(mini_gpt_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_inference PUBLIC mini_gpt_gpt)
target_compile_features(mini_gpt_inference PUBLIC cxx_std_20)
set_target_properties(mini_gpt_inference PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_inference PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_backpropagation STATIC
    src/backpropagation.cpp
)

target_include_directories(mini_gpt_backpropagation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_backpropagation
    PUBLIC
        mini_gpt_gpt
        mini_gpt_loss
)
target_compile_features(mini_gpt_backpropagation PUBLIC cxx_std_20)
set_target_properties(mini_gpt_backpropagation PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_backpropagation PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_backpropagation PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_optimizer STATIC
    src/adamw.cpp
)

target_include_directories(mini_gpt_optimizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_optimizer PUBLIC mini_gpt_backpropagation)
target_compile_features(mini_gpt_optimizer PUBLIC cxx_std_20)
set_target_properties(mini_gpt_optimizer PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_optimizer PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_optimizer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_data STATIC
    src/data_loader.cpp
)

target_include_directories(mini_gpt_data
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_data PUBLIC mini_gpt_tokenizer)
target_compile_features(mini_gpt_data PUBLIC cxx_std_20)
set_target_properties(mini_gpt_data PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_data PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_data PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_cli STATIC
    src/cli.cpp
    src/chat_mode.cpp
)

target_include_directories(mini_gpt_cli
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(mini_gpt_cli PUBLIC cxx_std_20)
set_target_properties(mini_gpt_cli PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_cli PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(tokenizer_train
    apps/tokenizer_train.cpp
)

target_link_libraries(tokenizer_train PRIVATE mini_gpt_cli mini_gpt_data)
target_compile_features(tokenizer_train PRIVATE cxx_std_20)
set_target_properties(tokenizer_train PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(tokenizer_train PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(tokenizer_train PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(mini_gpt_training_state STATIC
    src/training_state.cpp
)

target_include_directories(mini_gpt_training_state
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_gpt_training_state PUBLIC mini_gpt_data)
target_compile_features(mini_gpt_training_state PUBLIC cxx_std_20)
set_target_properties(mini_gpt_training_state PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(mini_gpt_training_state PRIVATE /W4 /permissive-)
else()
    target_compile_options(mini_gpt_training_state PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gpt_train
    apps/gpt_train.cpp
)

target_link_libraries(gpt_train
    PRIVATE
        mini_gpt_cli
        mini_gpt_training_state
        mini_gpt_optimizer
        mini_gpt_tokenizer
)
target_compile_features(gpt_train PRIVATE cxx_std_20)
set_target_properties(gpt_train PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(gpt_train PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(gpt_train PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gpt_chat
    apps/gpt_chat.cpp
)

target_link_libraries(gpt_chat PRIVATE mini_gpt_cli mini_gpt_inference mini_gpt_gpt mini_gpt_tokenizer)
target_compile_features(gpt_chat PRIVATE cxx_std_20)
set_target_properties(gpt_chat PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(gpt_chat PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(gpt_chat PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(corpus_prepare
    apps/corpus_prepare.cpp
)

target_link_libraries(corpus_prepare PRIVATE mini_gpt_data mini_gpt_gpt)
target_compile_features(corpus_prepare PRIVATE cxx_std_20)
set_target_properties(corpus_prepare PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(corpus_prepare PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(corpus_prepare PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(corpus_training_demo
    apps/corpus_training_demo.cpp
)

target_link_libraries(corpus_training_demo
    PRIVATE
        mini_gpt_data
        mini_gpt_optimizer
        mini_gpt_tokenizer
)
target_compile_features(corpus_training_demo PRIVATE cxx_std_20)
set_target_properties(corpus_training_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(corpus_training_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(corpus_training_demo PRIVATE -Wall -Wextra -Wpedantic)
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

add_executable(transformer_demo
    apps/transformer_demo.cpp
)

target_link_libraries(transformer_demo
    PRIVATE
        mini_gpt_transformer_block
        mini_gpt_embedding
        mini_gpt_tokenizer
)
target_compile_features(transformer_demo PRIVATE cxx_std_20)
set_target_properties(transformer_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(transformer_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(transformer_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gpt_demo
    apps/gpt_demo.cpp
)

target_link_libraries(gpt_demo
    PRIVATE
        mini_gpt_gpt
        mini_gpt_tokenizer
)
target_compile_features(gpt_demo PRIVATE cxx_std_20)
set_target_properties(gpt_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(gpt_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(gpt_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(loss_demo
    apps/loss_demo.cpp
)

target_link_libraries(loss_demo
    PRIVATE
        mini_gpt_loss
        mini_gpt_gpt
        mini_gpt_tokenizer
)
target_compile_features(loss_demo PRIVATE cxx_std_20)
target_compile_definitions(loss_demo PRIVATE
    MINI_GPT_LOSS_CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/conversations.txt"
)
set_target_properties(loss_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(loss_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(loss_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(backprop_demo
    apps/backprop_demo.cpp
)

target_link_libraries(backprop_demo
    PRIVATE
        mini_gpt_backpropagation
        mini_gpt_tokenizer
)
target_compile_features(backprop_demo PRIVATE cxx_std_20)
target_compile_definitions(backprop_demo PRIVATE
    MINI_GPT_BACKPROP_CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/conversations.txt"
)
set_target_properties(backprop_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(backprop_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(backprop_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(benchmark_demo
    apps/benchmark_demo.cpp
)

target_link_libraries(benchmark_demo
    PRIVATE
        mini_gpt_optimizer
        mini_gpt_data
        mini_gpt_tokenizer
)
target_compile_features(benchmark_demo PRIVATE cxx_std_20)
set_target_properties(benchmark_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(benchmark_demo PRIVATE /W4 /permissive-)
else()
    target_compile_options(benchmark_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(training_demo
    apps/training_demo.cpp
)

target_link_libraries(training_demo
    PRIVATE
        mini_gpt_optimizer
        mini_gpt_tokenizer
)
target_compile_features(training_demo PRIVATE cxx_std_20)
target_compile_definitions(training_demo PRIVATE
    MINI_GPT_TRAINING_CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/conversations.txt"
)
set_target_properties(training_demo PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(training_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(training_demo PRIVATE -Wall -Wextra -Wpedantic)
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

    add_executable(transformer_block_tests
        tests/transformer_block_tests.cpp
    )

    target_link_libraries(transformer_block_tests PRIVATE mini_gpt_transformer_block)
    set_target_properties(transformer_block_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(transformer_block_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(transformer_block_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME transformer_block_tests COMMAND transformer_block_tests)

    add_executable(gpt_tests
        tests/gpt_tests.cpp
    )

    target_link_libraries(gpt_tests PRIVATE mini_gpt_gpt)
    set_target_properties(gpt_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(gpt_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(gpt_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME gpt_tests COMMAND gpt_tests)

    add_executable(loss_tests
        tests/loss_tests.cpp
    )

    target_link_libraries(loss_tests PRIVATE mini_gpt_loss)
    set_target_properties(loss_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(loss_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(loss_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME loss_tests COMMAND loss_tests)

    add_executable(backpropagation_tests
        tests/backpropagation_tests.cpp
    )

    target_link_libraries(backpropagation_tests PRIVATE mini_gpt_backpropagation)
    set_target_properties(backpropagation_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(backpropagation_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(backpropagation_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME backpropagation_tests COMMAND backpropagation_tests)

    add_executable(optimizer_tests
        tests/optimizer_tests.cpp
    )

    target_link_libraries(optimizer_tests PRIVATE mini_gpt_optimizer)
    set_target_properties(optimizer_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(optimizer_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(optimizer_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME optimizer_tests COMMAND optimizer_tests)

    add_executable(data_loader_tests
        tests/data_loader_tests.cpp
    )

    target_link_libraries(data_loader_tests PRIVATE mini_gpt_data mini_gpt_optimizer)
    set_target_properties(data_loader_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(data_loader_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(data_loader_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME data_loader_tests COMMAND data_loader_tests)

    add_executable(cli_tests
        tests/cli_tests.cpp
    )

    target_link_libraries(cli_tests PRIVATE mini_gpt_cli mini_gpt_training_state)
    set_target_properties(cli_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(cli_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cli_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME cli_tests COMMAND cli_tests)

    add_executable(inference_tests
        tests/inference_tests.cpp
    )

    target_link_libraries(inference_tests PRIVATE mini_gpt_inference)
    set_target_properties(inference_tests PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(inference_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(inference_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME inference_tests COMMAND inference_tests)
endif()
