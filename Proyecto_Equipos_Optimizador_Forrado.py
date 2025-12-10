import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# CONFIGURACIÓN Y DATOS DEL NEGOCIO
menu_titulo = "*** Menú Forrado de Libros y Cuadernos ***"
elementos = [] 
tamanos = ["Bolsillo", "A7", "A6", "A5", "A4", "Folio", "Tapa Dura", "De Arte"]
medidas_tamanos = {
    "Bolsillo": [(11, 17), (15, 21)],
    "A7": [(7.4, 10.5)],
    "A6": [(10.5, 14.8)],
    "A5": [(14.8, 21)],
    "A4": [(21, 29.7)],
    "Folio": [(21.5, 31.5)],
    "Tapa Dura": [(15, 23)],
    "De Arte": [(30, 30)]
}

# ==========================================


# ESTRUCTURAS DE DATOS

@dataclass
class HojaMaterial:
    ancho: float
    alto: float

@dataclass
class TipoElemento:
    id: int
    ancho: float
    alto: float
    cantidad: int
    nombre: str
    lomo: float = 0.0 # Guardamos el grosor del lomo

@dataclass
class Elemento:
    id_tipo: int
    ancho: float
    alto: float
    nombre: str
    lomo: float = 0.0 
    x: float = 0.0
    y: float = 0.0
    rotado: bool = False # Para saber cómo dibujarlo

@dataclass
class TPila:
    elementos: List[Elemento] = field(default_factory=list)
    ancho: float = 0.0
    alto_usado: float = 0.0

@dataclass
class TTira:
    pilas: List[TPila] = field(default_factory=list)
    alto: float = 0.0
    ancho_usado: float = 0.0

@dataclass
class THojaCortada:
    tiras: List[TTira] = field(default_factory=list)
    alto_usado: float = 0.0

@dataclass
class Individuo:
    genes: List[int] 
    cortado: List[THojaCortada] = field(default_factory=list)
    aptitud: float = float('inf')

    def __lt__(self, other):
        return self.aptitud < other.aptitud

# MOTOR EVOLUTIVO (LÓGICA)

class AlgoritmoEvolutivo:
    
    def __init__(self, tam_poblacion=50, prob_cruce=0.7, prob_mutacion=0.01, num_generaciones=150):
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.num_generaciones = num_generaciones
        self.hoja_base: HojaMaterial = None
        self.tipos_elementos: List[TipoElemento] = []
        self.mapa_tipos: Dict[int, TipoElemento] = {}
        self.poblacion: List[Individuo] = []

    def cargar_datos(self, hoja: HojaMaterial, elementos_procesados: List[Dict]):
        self.hoja_base = hoja
        self.tipos_elementos = []
        self.mapa_tipos = {}
        
        id_counter = 0
        for item in elementos_procesados:
            t = TipoElemento(
                id=id_counter,
                ancho=item['ancho'],
                alto=item['alto'],
                cantidad=1,
                nombre=item['nombre'],
                lomo=item['lomo'] # Cargamos el lomo
            )
            self.tipos_elementos.append(t)
            self.mapa_tipos[id_counter] = t
            id_counter += 1

    def generar_poblacion_inicial(self):
        indices_tipos = [t.id for t in self.tipos_elementos]
        self.poblacion = []
        for _ in range(self.tam_poblacion):
            genes = random.sample(indices_tipos, len(indices_tipos))
            self.poblacion.append(Individuo(genes=genes))

    def calcular_aptitud(self, individuo: Individuo):
        piezas_a_colocar = []
        for id_tipo in individuo.genes:
            tipo = self.mapa_tipos[id_tipo]
            # Pasamos el dato del lomo al elemento
            piezas_a_colocar.append(Elemento(tipo.id, tipo.ancho, tipo.alto, tipo.nombre, lomo=tipo.lomo))
        
        W_HOJA = self.hoja_base.ancho
        L_HOJA = self.hoja_base.alto
        
        patron_cortado = [THojaCortada()]
        hoja_actual = patron_cortado[-1]
        
        for elem in piezas_a_colocar:
            colocado = False
            
            # Orientaciones: (Ancho, Alto, EsRotado)
            orientaciones = [(elem.ancho, elem.alto, False)]
            if elem.ancho != elem.alto:
                orientaciones.append((elem.alto, elem.ancho, True))
            
            for w, h, es_rotado in orientaciones:
                # Si rotamos, actualizamos el nombre para depuración
                nombre_uso = elem.nombre
                
                # --- LÓGICA DE ACOMODO ---
                # 1. Pila actual
                if hoja_actual.tiras:
                    tira = hoja_actual.tiras[-1]
                    if tira.pilas:
                        pila = tira.pilas[-1]
                        if w == pila.ancho and h + pila.alto_usado <= tira.alto:
                            nuevo = Elemento(elem.id_tipo, w, h, nombre_uso, lomo=elem.lomo, rotado=es_rotado)
                            pila.elementos.append(nuevo)
                            pila.alto_usado += h
                            colocado = True; break
                
                # 2. Nueva Pila en Tira actual
                if not colocado and hoja_actual.tiras:
                    tira = hoja_actual.tiras[-1]
                    if w + tira.ancho_usado <= W_HOJA and h <= tira.alto:
                        nuevo = Elemento(elem.id_tipo, w, h, nombre_uso, lomo=elem.lomo, rotado=es_rotado)
                        nueva_pila = TPila(ancho=w, alto_usado=h)
                        nueva_pila.elementos.append(nuevo)
                        tira.pilas.append(nueva_pila)
                        tira.ancho_usado += w
                        colocado = True; break
                
                # 3. Nueva Tira en Hoja actual
                if not colocado:
                    if w <= W_HOJA and h + hoja_actual.alto_usado <= L_HOJA:
                        nuevo = Elemento(elem.id_tipo, w, h, nombre_uso, lomo=elem.lomo, rotado=es_rotado)
                        nueva_pila = TPila(ancho=w, alto_usado=h, elementos=[nuevo])
                        nueva_tira = TTira(alto=h, ancho_usado=w, pilas=[nueva_pila])
                        hoja_actual.tiras.append(nueva_tira)
                        hoja_actual.alto_usado += h
                        colocado = True; break
            
            # 4. Nueva Hoja
            if not colocado:
                w, h = elem.ancho, elem.alto
                rot = False
                if w > W_HOJA or h > L_HOJA:
                    if h <= W_HOJA and w <= L_HOJA: 
                        w, h = h, w
                        rot = True
                    else: continue 

                hoja_actual = THojaCortada()
                patron_cortado.append(hoja_actual)
                
                nuevo = Elemento(elem.id_tipo, w, h, elem.nombre, lomo=elem.lomo, rotado=rot)
                nueva_pila = TPila(ancho=w, alto_usado=h, elementos=[nuevo])
                nueva_tira = TTira(alto=h, ancho_usado=w, pilas=[nueva_pila])
                hoja_actual.tiras.append(nueva_tira)
                hoja_actual.alto_usado = h

        individuo.cortado = patron_cortado
        num_hojas = len(patron_cortado)
        
        if num_hojas == 0:
            individuo.aptitud = float('inf')
        else:
            ultimo_uso = patron_cortado[-1].alto_usado / L_HOJA
            individuo.aptitud = num_hojas + (1.0 - ultimo_uso)

    def cruce(self, p1: Individuo, p2: Individuo):
        size = len(p1.genes)
        if size < 2: return copy.deepcopy(p1), copy.deepcopy(p2)
        a, b = sorted(random.sample(range(size), 2))
        
        def generar_hijo(base, aporte):
            hijo_genes = [-1] * size
            hijo_genes[a:b+1] = base.genes[a:b+1]
            pos = (b + 1) % size
            for gen in aporte.genes[b+1:] + aporte.genes[:b+1]:
                if gen not in hijo_genes:
                    hijo_genes[pos] = gen
                    pos = (pos + 1) % size
            return Individuo(genes=hijo_genes)

        return generar_hijo(p1, p2), generar_hijo(p2, p1)

    def mutacion(self, ind: Individuo):
        if random.random() < self.prob_mutacion:
            idx1, idx2 = random.sample(range(len(ind.genes)), 2)
            ind.genes[idx1], ind.genes[idx2] = ind.genes[idx2], ind.genes[idx1]

    def ejecutar(self):
        self.generar_poblacion_inicial()
        mejor_global = self.poblacion[0]
        
        print("\n--- Analizando combinaciones ---")
        for gen in range(self.num_generaciones):
            for ind in self.poblacion:
                self.calcular_aptitud(ind)
                if ind.aptitud < mejor_global.aptitud:
                    mejor_global = copy.deepcopy(ind)
            
            nueva_pob = [mejor_global]
            while len(nueva_pob) < self.tam_poblacion:
                p1 = min(random.sample(self.poblacion, 3), key=lambda x: x.aptitud)
                p2 = min(random.sample(self.poblacion, 3), key=lambda x: x.aptitud)
                if random.random() < self.prob_cruce:
                    h1, h2 = self.cruce(p1, p2)
                else:
                    h1, h2 = copy.deepcopy(p1), copy.deepcopy(p2)
                self.mutacion(h1)
                self.mutacion(h2)
                nueva_pob.extend([h1, h2])
            
            self.poblacion = nueva_pob[:self.tam_poblacion]
            if gen % 10 == 0:
                print(f"Progreso {gen}/{self.num_generaciones}...")

        return mejor_global

# VISUALIZACIÓN DETALLADA

def dibujar_libro_interno(ax, x, y, w, h, lomo, rotado, nombre):
    """ Dibuja las tapas, lomo y pestañas dentro del rectángulo asignado """
    pestana = 2.0
    
    # Colores
    c_pestana = '#A9DFBF' # Verde suave
    c_tapa1 = '#5DADE2'   # Azul
    c_tapa2 = '#F5B041'   # Naranja
    c_lomo = '#566573'    # Gris oscuro
    
    # Dibujar el fondo completo como pestaña (el margen)
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor='black', facecolor=c_pestana))
    
    if not rotado:  
        # --- ORIENTACIÓN NORMAL (Lomo Vertical) ---
        # w total = pest + tapa1 + lomo + tapa2 + pest
        # h total = pest + alto_tapa + pest
        
        ancho_util = w - (2 * pestana)
        alto_util = h - (2 * pestana)
        
        if ancho_util > 0 and alto_util > 0:
            ancho_tapa = (ancho_util - lomo) / 2
            
            # Coordenadas internas
            x_curr = x + pestana
            y_curr = y + pestana
            
            # Tapa 1 (Azul)
            ax.add_patch(patches.Rectangle((x_curr, y_curr), ancho_tapa, alto_util, 
                                         linewidth=0.5, edgecolor='black', facecolor=c_tapa1, alpha=0.9))
            # Lomo (Gris)
            x_curr += ancho_tapa
            ax.add_patch(patches.Rectangle((x_curr, y_curr), lomo, alto_util, 
                                         linewidth=0.5, edgecolor='black', facecolor=c_lomo, alpha=0.9))
            # Tapa 2 (Naranja)
            x_curr += lomo
            ax.add_patch(patches.Rectangle((x_curr, y_curr), ancho_tapa, alto_util, 
                                         linewidth=0.5, edgecolor='black', facecolor=c_tapa2, alpha=0.9))
            
            # Texto
            if ancho_tapa > 2:
                ax.text(x + w/2, y + h/2, nombre, ha='center', va='center', fontsize=7, color='white', fontweight='bold', rotation=0)

    else:
        ancho_util = w - (2 * pestana) # Esto es visualmente el ancho (altura del libro)
        alto_util = h - (2 * pestana)  # Esto es visualmente el alto (desplegado del libro)
        
        if ancho_util > 0 and alto_util > 0:
            alto_tapa = (alto_util - lomo) / 2
            
            x_curr = x + pestana
            y_curr = y + pestana
            
            # Tapa 1
            ax.add_patch(patches.Rectangle((x_curr, y_curr), ancho_util, alto_tapa, 
                                         linewidth=0.5, edgecolor='black', facecolor=c_tapa1, alpha=0.9))
            
            # Lomo (Horizontal ahora)
            y_curr += alto_tapa
            ax.add_patch(patches.Rectangle((x_curr, y_curr), ancho_util, lomo, 
                                         linewidth=0.5, edgecolor='black', facecolor=c_lomo, alpha=0.9))
            
            # Tapa 2
            y_curr += lomo
            ax.add_patch(patches.Rectangle((x_curr, y_curr), ancho_util, alto_tapa, 
                                         linewidth=0.5, edgecolor='black', facecolor=c_tapa2, alpha=0.9))
            
            # Texto rotado
            if alto_tapa > 2:
                ax.text(x + w/2, y + h/2, nombre, ha='center', va='center', fontsize=7, color='white', fontweight='bold', rotation=90)

def dibujar_resultado(hoja_base: HojaMaterial, hojas_cortadas: List[THojaCortada]):
    num_hojas = len(hojas_cortadas)
    if num_hojas == 0: 
        print("Nada que dibujar.")
        return

    # Ajustar tamaño figura
    fig, axs = plt.subplots(1, num_hojas, figsize=(6 * num_hojas, 8))
    if num_hojas == 1: axs = [axs]

    for i, (ax, hoja) in enumerate(zip(axs, hojas_cortadas)):
        ax.set_title(f"Papel Contact {i + 1}")
        ax.set_xlim(0, hoja_base.ancho)
        ax.set_ylim(0, hoja_base.alto)
        ax.set_aspect('equal')
        
        # Fondo hoja
        ax.add_patch(patches.Rectangle((0,0), hoja_base.ancho, hoja_base.alto, 
                                     edgecolor='black', facecolor='#f9f9f9', linewidth=2))
        
        y_tira = 0
        for tira in hoja.tiras:
            x_pila = 0
            for pila in tira.pilas:
                y_elem = y_tira
                for elem in pila.elementos:
                    dibujar_libro_interno(ax, x_pila, y_elem, elem.ancho, elem.alto, elem.lomo, elem.rotado, elem.nombre)
                    y_elem += elem.alto
                x_pila += pila.ancho
            y_tira += tira.alto
            
    plt.tight_layout()
    plt.show()



# INTEGRACIÓN LÓGICA DE NEGOCIO

def calcular_dimensiones_reales(item):
    ancho_tapa, alto_tapa = item["medidas_finales"]
    pestaña = 2.0
    
    ancho_lomo = 2.0 # Default
    if "personalizado" in item["encuadernación"]:
        try:
            str_val = item["encuadernación"].split('(')[1].split('x')[0]
            ancho_lomo = float(str_val)
        except: pass
        
    ancho_total = pestaña + ancho_tapa + ancho_lomo + ancho_tapa + pestaña
    alto_total = pestaña + alto_tapa + pestaña
    
    nombre_corto = f"{item['tipo'][0:3]}. {item['tamaño']}"
    
    return {
        "ancho": round(ancho_total, 2),
        "alto": round(alto_total, 2),
        "nombre": nombre_corto,
        "lomo": ancho_lomo
    }

def ejecutar_optimizacion():
    if not elementos:
        print("No hay elementos para forrar.")
        return

    try:
        print("\n--- Configuración del Papel ---")
        w = float(input("Ancho del rollo/hoja (cm): "))
        h = float(input("Alto del rollo/hoja (cm): "))
        
        hoja_base = HojaMaterial(w, h)
        
        piezas_procesadas = []
        for item in elementos:
            dims = calcular_dimensiones_reales(item)
            piezas_procesadas.append(dims)
            
        print("\nInicializando Inteligencia Artificial...")
        ae = AlgoritmoEvolutivo(tam_poblacion=60, num_generaciones=80)
        ae.cargar_datos(hoja_base, piezas_procesadas)
        
        mejor_solucion = ae.ejecutar()
        
        if mejor_solucion and mejor_solucion.cortado:
            print(f"\n¡Solución Encontrada!")
            print(f"Total hojas necesarias: {len(mejor_solucion.cortado)}")
            dibujar_resultado(hoja_base, mejor_solucion.cortado)
        else:
            print("No se encontró una solución válida.")

    except ValueError:
        print("Error en los números ingresados.")
        time.sleep(2)



# MENÚ DE USUARIO

def mostrar_lista():
    if elementos:
        print("\nLista de elementos a forrar:\n")
        print(f"{'#':<3} | {'Tipo':<10} | {'Tamaño':<10} | {'Encuadernación'}")
        print("-" * 50)
        for idx, item in enumerate(elementos, 1):
            print(f"{idx:<3} | {item['tipo']:<10} | {item['tamaño']:<10} | {item['encuadernación']}")
    else:
        print("\n(No hay elementos en la lista)")

def seleccionar_tamano():
    print("\nTamaños disponibles:")
    for i, t in enumerate(tamanos, 1):
        meds = medidas_tamanos[t]
        str_med = ", ".join([f"{m[0]}x{m[1]}" for m in meds])
        print(f"{i}. {t} ({str_med})")
    
    try:
        op = int(input("Elige número: "))
        if 1 <= op <= len(tamanos):
            nombre = tamanos[op-1]
            medidas = medidas_tamanos[nombre]
            seleccion = medidas[0]
            if len(medidas) > 1:
                print("Varias medidas disponibles:")
                for j, m in enumerate(medidas, 1): print(f"{j}. {m[0]}x{m[1]}")
                op_m = int(input("Cual: "))
                seleccion = medidas[op_m-1]
            return nombre, seleccion
    except: pass
    return tamanos[4], medidas_tamanos["A4"][0]

def agregar_elemento(tipo):
    tam_nombre, med = seleccionar_tamano()
    
    enc = "Espiral"
    print("1. Espiral\n2. Lomo personalizado")
    op_enc = input("> ")
    if op_enc == "2":
        grosor = input("Grosor lomo (cm): ")
        enc = f"personalizado ({grosor}x{med[1]})"
    else:
        enc = f"espiral ({2}x{med[1]})"
        
    cant = int(input("Cantidad: "))
    for _ in range(max(1, cant)):
        elementos.append({
            "tipo": tipo,
            "tamaño": tam_nombre,
            "encuadernación": enc,
            "medidas_finales": med
        })
    print("Elemento(s) agregado(s).")

def main():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(menu_titulo)
        mostrar_lista()
        print("\n1. Agregar Libro")
        print("2. Agregar Cuaderno")
        print("3. Simular acomodo en papel contact")
        print("4. Borrar lista")
        print("0. Salir")
        
        op = input("\nOpción: ")
        
        if op == "1": agregar_elemento("Libro")
        elif op == "2": agregar_elemento("Cuaderno")
        elif op == "3": ejecutar_optimizacion(); input("Enter para continuar...")
        elif op == "4": elementos.clear()
        elif op == "0": break

if __name__ == "__main__":
    main()