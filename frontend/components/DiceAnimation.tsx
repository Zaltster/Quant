import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import * as CANNON from 'cannon-es';

const DiceAnimation = (): React.ReactElement => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isRolling, setIsRolling] = useState(false);
    const animationRef = useRef<number | null>(null);
    const diceRef = useRef<Array<{ mesh: THREE.Mesh, body: CANNON.Body }>>([]);
    const [showButton, setShowButton] = useState(false);
    const worldRef = useRef<CANNON.World | null>(null);

    // Initialize scene and physics
    useEffect(() => {
        if (!containerRef.current) return;

        const container = containerRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Initialize Three.js scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(5, 10, 7);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 1024;
        directionalLight.shadow.mapSize.height = 1024;
        scene.add(directionalLight);

        // Setup camera
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 8, 14);
        camera.lookAt(0, 0, 0);

        // Initialize renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Add controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableZoom = false;
        controls.enablePan = false;
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Initialize physics
        const world = new CANNON.World();
        world.gravity.set(0, -9.82 * 3, 0); // Increased gravity for faster settling
        world.broadphase = new CANNON.NaiveBroadphase();
        world.defaultContactMaterial.friction = 0.3;
        worldRef.current = world;

        // Create materials
        const groundMaterial = new CANNON.Material("groundMaterial");
        const diceMaterial = new CANNON.Material("diceMaterial");

        // Define contact properties
        const contact = new CANNON.ContactMaterial(
            groundMaterial,
            diceMaterial,
            {
                friction: 0.4,
                restitution: 0.3,
            }
        );

        world.addContactMaterial(contact);

        // Create ground plane
        const groundShape = new CANNON.Plane();
        const groundBody = new CANNON.Body({
            mass: 0, // static body
            material: groundMaterial,
            shape: groundShape
        });
        groundBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0); // rotate to be flat
        groundBody.position.set(0, -0.5, 0);
        world.addBody(groundBody);

        // Create table
        const tableGeometry = new THREE.BoxGeometry(10, 0.5, 10);
        const tableMaterial = new THREE.MeshStandardMaterial({
            color: 0x1e6b24, // Green felt color
            roughness: 0.8,
            metalness: 0.2
        });
        const table = new THREE.Mesh(tableGeometry, tableMaterial);
        table.position.y = -0.5;
        table.receiveShadow = true;
        scene.add(table);

        // Create walls
        createWall(world, scene, 10, 2, 0.2, 0, 0, -5, groundMaterial);
        createWall(world, scene, 10, 2, 0.2, 0, 0, 5, groundMaterial);
        createWall(world, scene, 0.2, 2, 10, -5, 0, 0, groundMaterial);
        createWall(world, scene, 0.2, 2, 10, 5, 0, 0, groundMaterial);

        // Animation loop
        const animate = () => {
            animationRef.current = requestAnimationFrame(animate);

            // Update physics
            if (world) {
                world.step(1 / 60);

                // Update dice position
                diceRef.current.forEach(dice => {
                    const position = dice.body.position;
                    const quaternion = dice.body.quaternion;

                    dice.mesh.position.set(position.x, position.y, position.z);
                    dice.mesh.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
                });

                // Check if dice have settled
                if (isRolling) {
                    const allSettled = diceRef.current.every(dice => {
                        const velocity = dice.body.velocity;
                        const angularVelocity = dice.body.angularVelocity;

                        return (
                            Math.abs(velocity.x) < 0.1 &&
                            Math.abs(velocity.y) < 0.1 &&
                            Math.abs(velocity.z) < 0.1 &&
                            Math.abs(angularVelocity.x) < 0.1 &&
                            Math.abs(angularVelocity.y) < 0.1 &&
                            Math.abs(angularVelocity.z) < 0.1
                        );
                    });

                    if (allSettled) {
                        setIsRolling(false);
                        setShowButton(true);
                    }
                }
            }

            controls.update();
            renderer.render(scene, camera);
        };

        animate();

        // Auto-roll on first render
        setTimeout(() => {
            createDice(world, scene, -2, 5, -2);
            createDice(world, scene, 2, 5, 2);
        }, 500);

        // Cleanup on unmount
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }

            if (renderer) {
                container.removeChild(renderer.domElement);
            }
        };
    }, []);

    // Create a physics wall
    const createWall = (
        world: CANNON.World,
        scene: THREE.Scene,
        width: number,
        height: number,
        depth: number,
        x: number,
        y: number,
        z: number,
        material: CANNON.Material
    ) => {
        // Physics body
        const halfExtents = new CANNON.Vec3(width / 2, height / 2, depth / 2);
        const wallShape = new CANNON.Box(halfExtents);
        const wallBody = new CANNON.Body({
            mass: 0,
            material: material,
            shape: wallShape
        });
        wallBody.position.set(x, y, z);
        world.addBody(wallBody);

        // Visual mesh
        const wallMaterial = new THREE.MeshStandardMaterial({
            color: 0x654321, // Brown color
            roughness: 0.8,
            metalness: 0.1
        });

        const wallGeometry = new THREE.BoxGeometry(width, height, depth);
        const wall = new THREE.Mesh(wallGeometry, wallMaterial);
        wall.position.set(x, y, z);
        wall.receiveShadow = true;
        wall.castShadow = true;
        scene.add(wall);
    };

    // Create a die
    const createDice = (world: CANNON.World, scene: THREE.Scene, x: number, y: number, z: number) => {
        // Create die geometry
        const dieSize = 1;
        const geometry = new THREE.BoxGeometry(dieSize, dieSize, dieSize);
        const material = new THREE.MeshStandardMaterial({
            color: 0xdc143c, // Red color
            roughness: 0.5,
            metalness: 0.1
        });

        // Create die mesh
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        // Add die dots
        addDieDots(mesh);

        // Create physics body
        const diceMaterial = new CANNON.Material("diceMaterial");
        const shape = new CANNON.Box(new CANNON.Vec3(dieSize / 2, dieSize / 2, dieSize / 2));
        const body = new CANNON.Body({
            mass: 1,
            material: diceMaterial,
            shape: shape,
        });

        // Set initial position and velocity
        body.position.set(x, y, z);
        // Set random initial angular velocity for spinning
        body.angularVelocity.set(
            Math.random() * 20 - 10,
            Math.random() * 20 - 10,
            Math.random() * 20 - 10
        );

        // Add to simulation
        world.addBody(body);

        // Apply impulse
        body.applyImpulse(
            new CANNON.Vec3(8 + Math.random() * 4, -2, 4 + Math.random() * 4),
            new CANNON.Vec3(body.position.x, body.position.y, body.position.z)
        );

        // Add to reference
        diceRef.current.push({ mesh, body });

        return { mesh, body };
    };

    // Add dots to a die face
    const addDieDots = (dieMesh: THREE.Mesh) => {
        const dotSize = 0.12;
        const dotGeometry = new THREE.SphereGeometry(dotSize, 16, 16);
        const dotMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff });

        // Face 1: 1 dot in center
        const dot1 = new THREE.Mesh(dotGeometry, dotMaterial);
        dot1.position.set(0, 0, 0.5 + 0.01);
        dieMesh.add(dot1);

        // Face 2: 2 dots diagonally
        const dot2a = new THREE.Mesh(dotGeometry, dotMaterial);
        dot2a.position.set(0.25, 0.25, -0.5 - 0.01);
        dieMesh.add(dot2a);

        const dot2b = new THREE.Mesh(dotGeometry, dotMaterial);
        dot2b.position.set(-0.25, -0.25, -0.5 - 0.01);
        dieMesh.add(dot2b);

        // Face 3: 3 dots diagonally (right side)
        const dot3a = new THREE.Mesh(dotGeometry, dotMaterial);
        dot3a.position.set(0.5 + 0.01, 0.25, 0.25);
        dieMesh.add(dot3a);

        const dot3b = new THREE.Mesh(dotGeometry, dotMaterial);
        dot3b.position.set(0.5 + 0.01, 0, 0);
        dieMesh.add(dot3b);

        const dot3c = new THREE.Mesh(dotGeometry, dotMaterial);
        dot3c.position.set(0.5 + 0.01, -0.25, -0.25);
        dieMesh.add(dot3c);

        // Face 4: 4 dots in corners (left side)
        const dot4a = new THREE.Mesh(dotGeometry, dotMaterial);
        dot4a.position.set(-0.5 - 0.01, 0.25, 0.25);
        dieMesh.add(dot4a);

        const dot4b = new THREE.Mesh(dotGeometry, dotMaterial);
        dot4b.position.set(-0.5 - 0.01, 0.25, -0.25);
        dieMesh.add(dot4b);

        const dot4c = new THREE.Mesh(dotGeometry, dotMaterial);
        dot4c.position.set(-0.5 - 0.01, -0.25, 0.25);
        dieMesh.add(dot4c);

        const dot4d = new THREE.Mesh(dotGeometry, dotMaterial);
        dot4d.position.set(-0.5 - 0.01, -0.25, -0.25);
        dieMesh.add(dot4d);

        // Face 5: 5 dots (4 corners + center) (top)
        const dot5a = new THREE.Mesh(dotGeometry, dotMaterial);
        dot5a.position.set(0.25, 0.5 + 0.01, 0.25);
        dieMesh.add(dot5a);

        const dot5b = new THREE.Mesh(dotGeometry, dotMaterial);
        dot5b.position.set(-0.25, 0.5 + 0.01, 0.25);
        dieMesh.add(dot5b);

        const dot5c = new THREE.Mesh(dotGeometry, dotMaterial);
        dot5c.position.set(0, 0.5 + 0.01, 0);
        dieMesh.add(dot5c);

        const dot5d = new THREE.Mesh(dotGeometry, dotMaterial);
        dot5d.position.set(0.25, 0.5 + 0.01, -0.25);
        dieMesh.add(dot5d);

        const dot5e = new THREE.Mesh(dotGeometry, dotMaterial);
        dot5e.position.set(-0.25, 0.5 + 0.01, -0.25);
        dieMesh.add(dot5e);

        // Face 6: 6 dots (2 rows of 3) (bottom)
        const dot6a = new THREE.Mesh(dotGeometry, dotMaterial);
        dot6a.position.set(0.25, -0.5 - 0.01, 0.25);
        dieMesh.add(dot6a);

        const dot6b = new THREE.Mesh(dotGeometry, dotMaterial);
        dot6b.position.set(0, -0.5 - 0.01, 0.25);
        dieMesh.add(dot6b);

        const dot6c = new THREE.Mesh(dotGeometry, dotMaterial);
        dot6c.position.set(-0.25, -0.5 - 0.01, 0.25);
        dieMesh.add(dot6c);

        const dot6d = new THREE.Mesh(dotGeometry, dotMaterial);
        dot6d.position.set(0.25, -0.5 - 0.01, -0.25);
        dieMesh.add(dot6d);

        const dot6e = new THREE.Mesh(dotGeometry, dotMaterial);
        dot6e.position.set(0, -0.5 - 0.01, -0.25);
        dieMesh.add(dot6e);

        const dot6f = new THREE.Mesh(dotGeometry, dotMaterial);
        dot6f.position.set(-0.25, -0.5 - 0.01, -0.25);
        dieMesh.add(dot6f);
    };

    // Roll dice
    const rollDice = () => {
        if (isRolling || !worldRef.current) return;

        setIsRolling(true);
        setShowButton(false);

        // Remove existing dice
        diceRef.current.forEach(die => {
            die.mesh.removeFromParent();
            worldRef.current?.removeBody(die.body);
        });
        diceRef.current = [];

        // Create new dice
        if (worldRef.current) {
            const scene = (worldRef.current as any)._threeScene || (diceRef.current[0]?.mesh.parent as THREE.Scene);
            createDice(worldRef.current, scene, -2, 5, -2);
            createDice(worldRef.current, scene, 2, 5, 2);
        }
    };

    return (
        <div className="relative w-full" style={{ height: '300px' }}>
            <div
                ref={containerRef}
                className="w-full h-full bg-transparent"
            />

            {showButton && !isRolling && (
                <button
                    onClick={rollDice}
                    className="absolute px-6 py-2 rounded-md font-bold text-white bg-blue-500 hover:bg-blue-600 transform transition-transform hover:scale-105"
                    style={{
                        bottom: '-40px',
                        left: '50%',
                        transform: 'translateX(-50%)'
                    }}
                >
                    Roll Dice
                </button>
            )}
        </div>
    );
};

export default DiceAnimation;