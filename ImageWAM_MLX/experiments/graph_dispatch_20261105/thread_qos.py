"""Experimental macOS current-thread QoS; changes only this test process."""
import ctypes
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
lib.pthread_self.argtypes=[];lib.pthread_self.restype=ctypes.c_void_p
lib.pthread_get_qos_class_np.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_int)]
lib.pthread_get_qos_class_np.restype=ctypes.c_int
lib.pthread_set_qos_class_self_np.argtypes=[ctypes.c_uint,ctypes.c_int];lib.pthread_set_qos_class_self_np.restype=ctypes.c_int
def get():
 q=ctypes.c_uint();p=ctypes.c_int();rc=lib.pthread_get_qos_class_np(lib.pthread_self(),ctypes.byref(q),ctypes.byref(p))
 if rc:raise OSError(rc,'pthread_get_qos_class_np failed')
 return dict(qos=q.value,relative_priority=p.value)
def apply(mode):
 before=get()
 if mode!='inherit':
  rc=lib.pthread_set_qos_class_self_np({'initiated':0x19,'interactive':0x21}[mode],0)
  if rc:raise OSError(rc,'pthread_set_qos_class_self_np failed')
 return dict(requested=mode,before=before,after=get())
