
# Тесты Level 2 BLAS через ctypes + OpenBLAS

## Как работает в двух словах

1. Находим файл библиотеки (.dll / .so)
2. Загружаем её: `lib = ctypes.CDLL(LIB_PATH)` — теперь можем звать функции OpenBLAS  
3. Говорим Python, какие аргументы ждут функции (setup_argtypes)  
4. В тестах создаём массивы в numpy → передаём адреса в OpenBLAS через ctypes → OpenBLAS считает → сравниваем результат с ожидаемым через numpy

## Поиск файла библиотеки
 ```python
    def find_openblas():
    lib_path = r"C:\Users\User\Downloads\OpenBLAS-0.3.31-x64\win64\bin\libopenblas.dll"
   ```
 Просто конкретный путь на скачанную библиотеку внутри функции, после присваивание результата функции к переменной.

```python
   LIB_PATH = find_openblas()
   ```

## Загрузка библиотеки

```python
lib = ctypes.CDLL(LIB_PATH)
```

Загружает OpenBLAS в память.  
После этого можно писать `lib.cblas_sgemv(...)` — Python позовёт нужную функцию из библиотеки.

## setup_argtypes() — настройка типов данных для передачи через ctypes

```python
def setup_argtypes():
    lib.cblas_sgemv.argtypes = [ctypes.c_int, ... , ctypes.c_float, ctypes.c_void_p, ...]
```

Главные типы:
- `c_int` — целые числа 
- `c_float` / `c_double` — обычные и точные дробные числа
- `c_void_p` — адрес в памяти

## Пример теста (test_sgemv)

```python
def test_sgemv(self):
    A = np.array([1,2,3,4], np.float32).reshape(2,2)
    x = np.array([1,1], np.float32)
    y = np.zeros(2, np.float32)
    expected = np.array([3,7], np.float32)

    lib.cblas_sgemv(..., A.ctypes.data_as(ctypes.c_void_p), ..., y.ctypes.data_as(ctypes.c_void_p), ...)
    
    np.testing.assert_allclose(y, expected, rtol=EPS_F, atol=EPS_F)
```

- numpy делает массивы c данными
- через ctypes отдаём их адреса OpenBLAS  
- OpenBLAS отрабатывает функцию и пишет в y  
- `np.testing.assert_allclose` проверяет: почти ли равно y тому, что ждали

Остальные тесты работают так же.

## Запуск тестов (if __name__ == '__main__')

```python
if __name__ == '__main__':
    print("CBLAS LEVEL 2 TESTS")
    print(f"Using OpenBLAS from: {LIB_PATH}")
    print()
    unittest.main(argv=[''], verbosity=2)
```

- `if __name__ == '__main__':` — срабатывает только когда запускаешь файл напрямую (`python tests.py`)   
- `unittest.main()` — запускает все тесты  
  Как он их находит?  
  Ищет все классы, имя которых начинается с `Test` (`TestCBLASLevel2`).  
  Внутри класса ищет функции, которые начинаются с `test_` (`test_sgemv`, `test_dgemv` и т.д.).  
  Всё, что нашёл — запускает по очереди.  
