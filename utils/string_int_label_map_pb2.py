#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: \
        x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()
#定义文件描述符
DESCRIPTOR = \
    _descriptor.FileDescriptor(name='object_detection/protos/string_int_label_map.proto'
                               , package='object_detection.protos',
                               serialized_pb=_b('''
2object_detection/protos/string_int_label_map.proto\x12\x17object_detection.protos\"G
\x15StringIntLabelMapItem\x12\x0c
\x04name\x18\x01 \x01(\t\x12

\x02id\x18\x02 \x01(\x05\x12\x14
\x0c\x64isplay_name\x18\x03 \x01(\t\"Q
\x11StringIntLabelMap\x12<
\x04item\x18\x01 \x03(\x0b\x32..object_detection.protos.StringIntLabelMapItem'''))
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

# 定义 StringIntLabelMapItem 的描述符
_STRINGINTLABELMAPITEM = _descriptor.Descriptor(
    name='StringIntLabelMapItem',
    full_name='object_detection.protos.StringIntLabelMapItem',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[_descriptor.FieldDescriptor(
        name='name',
        full_name='object_detection.protos.StringIntLabelMapItem.name',
        index=0,
        number=1,
        type=9,
        cpp_type=9,
        label=1,
        has_default_value=False,
        default_value=_b('').decode('utf-8'),
        message_type=None,
        enum_type=None,
        containing_type=None,
        is_extension=False,
        extension_scope=None,
        options=None,
        ), _descriptor.FieldDescriptor(
        name='id',
        full_name='object_detection.protos.StringIntLabelMapItem.id',
        index=1,
        number=2,
        type=5,
        cpp_type=1,
        label=1,
        has_default_value=False,
        default_value=0,
        message_type=None,
        enum_type=None,
        containing_type=None,
        is_extension=False,
        extension_scope=None,
        options=None,
        ), _descriptor.FieldDescriptor(
        name='display_name',
        full_name='object_detection.protos.StringIntLabelMapItem.display_name'
            ,
        index=2,
        number=3,
        type=9,
        cpp_type=9,
        label=1,
        has_default_value=False,
        default_value=_b('').decode('utf-8'),
        message_type=None,
        enum_type=None,
        containing_type=None,
        is_extension=False,
        extension_scope=None,
        options=None,
        )],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[],
    serialized_start=79,
    serialized_end=150,
    )
# 定义 StringIntLabelMap 的描述符
_STRINGINTLABELMAP = _descriptor.Descriptor(
    name='StringIntLabelMap',
    full_name='object_detection.protos.StringIntLabelMap',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[_descriptor.FieldDescriptor(
        name='item',
        full_name='object_detection.protos.StringIntLabelMap.item',
        index=0,
        number=1,
        type=11,
        cpp_type=10,
        label=3,
        has_default_value=False,
        default_value=[],
        message_type=None,
        enum_type=None,
        containing_type=None,
        is_extension=False,
        extension_scope=None,
        options=None,
        )],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[],
    serialized_start=152,
    serialized_end=233,
    )

_STRINGINTLABELMAP.fields_by_name['item'].message_type = \
    _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMapItem'] = \
    _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMap'] = \
    _STRINGINTLABELMAP

StringIntLabelMapItem = \
    _reflection.GeneratedProtocolMessageType('StringIntLabelMapItem',
        (_message.Message, ), dict(DESCRIPTOR=_STRINGINTLABELMAPITEM,
        __module__='object_detection.protos.string_int_label_map_pb2'))

  # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMapItem)

#注册 StringIntLabelMapItem 类到符号数据库
_sym_db.RegisterMessage(StringIntLabelMapItem)
#生成 StringIntLabelMap 类
StringIntLabelMap = \
    _reflection.GeneratedProtocolMessageType('StringIntLabelMap',
        (_message.Message, ), dict(DESCRIPTOR=_STRINGINTLABELMAP,
        __module__='object_detection.protos.string_int_label_map_pb2'))

  # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMap)
	
# 生成 StringIntLabelMap 到符号数据库
_sym_db.RegisterMessage(StringIntLabelMap)

# @@protoc_insertion_point(module_scope)............
