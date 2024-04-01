// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm80, half, uint8_t, 1, 128>>(const AttentionParams<half>&);

template bool invokeDecoding<Decoding<arch::Sm80, half, uint8_t, 4, 128>>(const AttentionParams<half>&);

template bool invokeDecoding<Decoding<arch::Sm80, half, uint8_t, 6, 128>>(const AttentionParams<half>&);

template bool invokeDecoding<Decoding<arch::Sm80, half, uint8_t, 8, 128>>(const AttentionParams<half>&);

}  // namespace turbomind
