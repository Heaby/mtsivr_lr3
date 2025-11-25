import struct
import os
import numpy as np
import sys

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL библиотеки не установлены")


def read_depth_map(filename):
    """Чтение карты глубины"""
    file_size = os.path.getsize(filename)
    print(f"Размер файла: {file_size} байт")

    with open(filename, 'rb') as f:
        # Чтение размеров как double
        height_raw = struct.unpack('d', f.read(8))[0]
        width_raw = struct.unpack('d', f.read(8))[0]

        height = int(round(height_raw))
        width = int(round(width_raw))

        print(f"Размеры из заголовка: {width} x {height}")

        if height <= 0 or width <= 0:
            raise ValueError(f"Некорректные размеры: {width} x {height}")

        # Чтение данных глубины
        data_size = width * height
        data = struct.unpack(f'{data_size}d', f.read(data_size * 8))

        print(f"Прочитано значений: {len(data)}")
        print(f"Диапазон глубин: {min(data):.3f} - {max(data):.3f}")

        return width, height, data


def create_3d_model_from_depth_map(depth_data, width, height):
    """Создание 3D модели"""
    vertices = []
    vertex_index_map = [-1] * (width * height)

    # Создание вершин
    for y in range(height):
        for x in range(width):
            grid_index = y * width + x
            depth = depth_data[grid_index]

            # Пропускаем точки с нулевой глубиной (фон)
            if depth > 0.0:
                point_x = float(x)
                point_y = float(height - y - 1)  # Инвертируем Y
                point_z = depth

                vertex_index_map[grid_index] = len(vertices)
                vertices.append((point_x, point_y, point_z))

    # Создание треугольных граней
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            idx1 = y * width + x
            idx2 = y * width + (x + 1)
            idx3 = (y + 1) * width + x
            idx4 = (y + 1) * width + (x + 1)

            v1 = vertex_index_map[idx1]
            v2 = vertex_index_map[idx2]
            v3 = vertex_index_map[idx3]
            v4 = vertex_index_map[idx4]

            # Проверяем, что все вершины существуют
            if v1 != -1 and v2 != -1 and v3 != -1 and v4 != -1:
                # Первый треугольник (v1, v2, v3)
                faces.append((v1, v2, v3))
                # Второй треугольник (v2, v4, v3)
                faces.append((v2, v4, v3))

    print(f"Создано вершин: {len(vertices)}")
    print(f"Создано треугольников: {len(faces)}")

    return vertices, faces


def export_to_vrml(vertices, faces, output_filename):
    """Экспорт в VRML формат"""
    with open(output_filename, 'w') as f:
        # Заголовок VRML
        f.write("#VRML V2.0 utf8\n")
        f.write("Shape {\n")
        f.write("  appearance Appearance {\n")
        f.write("    material Material {\n")
        f.write("      diffuseColor 0.8 0.8 0.8\n")
        f.write("      specularColor 0.5 0.5 0.5\n")
        f.write("      shininess 0.8\n")
        f.write("    }\n")
        f.write("  }\n")
        f.write("  geometry IndexedFaceSet {\n")
        f.write("    solid FALSE\n")
        f.write("    creaseAngle 0.5\n")

        # Запись координат вершин
        f.write("    coord Coordinate {\n")
        f.write("      point [\n")
        for x, y, z in vertices:
            f.write(f"        {x:.6f} {y:.6f} {z:.6f},\n")
        f.write("      ]\n")
        f.write("    }\n")

        # Запись индексов граней
        f.write("    coordIndex [\n")
        for face in faces:
            f.write(f"        {face[0]}, {face[1]}, {face[2]}, -1,\n")
        f.write("    ]\n")

        f.write("  }\n")
        f.write("}\n")


# ==================== OPENGL ВИЗУАЛИЗАЦИЯ ====================

if OPENGL_AVAILABLE:
    # Глобальные переменные для OpenGL
    gl_vertices = []
    gl_faces = []
    gl_rot_x = 55.0
    gl_rot_y = -45.0
    gl_zoom = 2.2
    gl_center = [0.0, 0.0, 0.0]
    gl_scale = 1.0


    def compute_normalization():
        """Вычисление центра и масштаба для нормализации"""
        global gl_center, gl_scale

        if not gl_vertices:
            gl_center = [0.0, 0.0, 0.0]
            gl_scale = 1.0
            return

        min_x = min(v[0] for v in gl_vertices)
        max_x = max(v[0] for v in gl_vertices)
        min_y = min(v[1] for v in gl_vertices)
        max_y = max(v[1] for v in gl_vertices)
        min_z = min(v[2] for v in gl_vertices)
        max_z = max(v[2] for v in gl_vertices)

        gl_center[0] = (min_x + max_x) * 0.5
        gl_center[1] = (min_y + max_y) * 0.5
        gl_center[2] = (min_z + max_z) * 0.5

        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        span_z = max(1e-6, max_z - min_z)
        max_span = max(span_x, span_y, span_z)
        gl_scale = 2.0 / max_span


    def setup_lights():
        """Настройка освещения с несколькими источниками для объема"""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)  # Добавляем второй источник света
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Основной источник света (спереди-сверху)
        light0_ambient = [0.2, 0.2, 0.2, 1.0]
        light0_diffuse = [0.8, 0.8, 0.8, 1.0]
        light0_position = [2.0, 3.0, 2.0, 0.0]  # Направленный свет

        glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light0_position)

        # Второй источник (заполняющий, для теней)
        light1_ambient = [0.1, 0.1, 0.1, 1.0]
        light1_diffuse = [0.4, 0.4, 0.4, 1.0]
        light1_position = [-1.0, -1.0, -1.0, 0.0]  # С противоположной стороны

        glLightfv(GL_LIGHT1, GL_AMBIENT, light1_ambient)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse)
        glLightfv(GL_LIGHT1, GL_POSITION, light1_position)

        # Включаем расчет теней
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    def draw_axes():
        """Отрисовка осей координат"""
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        # Y-axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        glEnable(GL_LIGHTING)


    def compute_vertex_normals():
        """Вычисление нормалей для вершин для лучшего освещения"""
        normals = [[0.0, 0.0, 0.0] for _ in range(len(gl_vertices))]

        for face in gl_faces:
            if len(face) == 3:
                v0, v1, v2 = face
                if all(0 <= idx < len(gl_vertices) for idx in [v0, v1, v2]):
                    p0 = gl_vertices[v0]
                    p1 = gl_vertices[v1]
                    p2 = gl_vertices[v2]

                    # Векторы сторон треугольника
                    u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
                    v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]

                    # Векторное произведение для нормали
                    normal = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0]
                    ]

                    # Нормализация
                    length = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
                    if length > 0:
                        normal = [n / length for n in normal]

                        # Добавляем нормаль ко всем вершинам треугольника
                        for vertex_idx in face:
                            normals[vertex_idx][0] += normal[0]
                            normals[vertex_idx][1] += normal[1]
                            normals[vertex_idx][2] += normal[2]

        # Нормализуем итоговые нормали
        for i in range(len(normals)):
            length = (normals[i][0] ** 2 + normals[i][1] ** 2 + normals[i][2] ** 2) ** 0.5
            if length > 0:
                normals[i] = [n / length for n in normals[i]]
            else:
                normals[i] = [0.0, 0.0, 1.0]

        return normals


    def draw_mesh():
        """Отрисовка 3D сетки с нормалями и освещением"""
        normals = compute_vertex_normals()

        # Материал с шероховатостью для лучшего восприятия объема
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.7, 0.7, 0.8, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 20.0)

        glColor3f(0.7, 0.7, 0.8)
        glBegin(GL_TRIANGLES)
        for face in gl_faces:
            if len(face) == 3:
                for vertex_index in face:
                    if 0 <= vertex_index < len(gl_vertices):
                        x, y, z = gl_vertices[vertex_index]
                        nx, ny, nz = normals[vertex_index]
                        glNormal3f(nx, ny, nz)
                        glVertex3f(x, y, z)
        glEnd()




    def display():
        """Функция отрисовки"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Настройка камеры
        glTranslatef(0.0, 0.0, float(-5.0 * gl_zoom))
        glRotatef(gl_rot_x, 1.0, 0.0, 0.0)
        glRotatef(gl_rot_y, 0.0, 1.0, 0.0)
        glScaled(gl_scale, gl_scale, gl_scale)
        glTranslated(-gl_center[0], -gl_center[1], -gl_center[2])

        # Отрисовка
        draw_axes()
        draw_mesh()

        glutSwapBuffers()


    def reshape(width, height):
        """Обработка изменения размера окна"""
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


    def keyboard(key, x, y):
        """Обработка клавиатуры"""
        global gl_rot_x, gl_rot_y, gl_zoom

        key = key.decode('utf-8').lower()

        if key in ['q', '\x1b']:  # Q или ESC
            sys.exit(0)
        elif key == 'w':
            gl_zoom = max(0.2, gl_zoom * 0.95)
        elif key == 's':
            gl_zoom = min(10.0, gl_zoom * 1.05)
        elif key == 'r':  # Reset view
            gl_rot_x = 55.0
            gl_rot_y = -45.0
            gl_zoom = 2.2

        glutPostRedisplay()


    def special_keys(key, x, y):
        """Обработка специальных клавиш"""
        global gl_rot_x, gl_rot_y

        step = 5.0
        if key == GLUT_KEY_LEFT:
            gl_rot_y -= step
        elif key == GLUT_KEY_RIGHT:
            gl_rot_y += step
        elif key == GLUT_KEY_UP:
            gl_rot_x -= step
        elif key == GLUT_KEY_DOWN:
            gl_rot_x += step

        glutPostRedisplay()


    def init_opengl():
        """Инициализация OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        setup_lights()


    def run_opengl_visualization(vertices, faces):
        """Запуск OpenGL визуализации"""
        global gl_vertices, gl_faces
        gl_vertices = vertices
        gl_faces = faces

        compute_normalization()

        # Инициализация GLUT
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(900, 600)
        glutCreateWindow(b"Depth Map Visualizer - Variant 12")

        init_opengl()

        # Регистрация callback-функций
        glutDisplayFunc(display)
        glutReshapeFunc(reshape)
        glutKeyboardFunc(keyboard)
        glutSpecialFunc(special_keys)

        print("\n Управление в OpenGL:")
        print("   Стрелки - вращение")
        print("   W/S - приближение/отдаление")
        print("   R - сброс вида")
        print("   Q/ESC - выход")

        glutMainLoop()


def main():
    """Основная программа"""
    input_filename = "DepthMap_12.dat"

    if not os.path.exists(input_filename):
        print(f"Файл {input_filename} не найден!")
        print("Убедитесь, что файл находится в той же папке, что и программа")
        return

    try:
        # 1. Чтение карты глубины
        width, height, depth_data = read_depth_map(input_filename)

        # 2. Создание 3D модели
        vertices, faces = create_3d_model_from_depth_map(depth_data, width, height)

        if not vertices:
            print("Нет данных для создания модели")
            return

        # 3. Экспорт в VRML с правильным расширением .vrml
        output_filename = "output12.vrml"
        export_to_vrml(vertices, faces, output_filename)

        print(f"Файл {output_filename} успешно создан")
        print(f"Результат: {len(vertices)} вершин, {len(faces)} треугольников")

        # 4. Визуализация в OpenGL для проверки
        if OPENGL_AVAILABLE:
            print("\nЗапуск OpenGL визуализации для проверки...")
            run_opengl_visualization(vertices, faces)
        else:
            print("\n  OpenGL не доступен для визуализации")

    except Exception as e:
        print(f" Ошибка: {e}")


if __name__ == "__main__":
    main()
