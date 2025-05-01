import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import * as CANNON from 'cannon-es';

const DiceAnimation = (): React.ReactElement => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isRolling, setIsRolling] = useState(false);
    const animationRef = useRef<number | null>(null);
    const diceRef = useRef<Array<{ mesh: THREE.Mesh, body: CANNON.Body, frozen: boolean, screenPosition: { x: number, y: number, z: number } }>>([]);
    const [showButton, setShowButton] = useState(false);
    const worldRef = useRef<CANNON.World | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

    // Initialize scene and physics
    useEffect(() => {
        if (!containerRef.current) return;

        const container = containerRef.current;

        // Get container dimensions
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Initialize Three.js scene
        const scene = new THREE.Scene();
        sceneRef.current = scene;
        // Make scene background transparent
        scene.background = null;

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); // Brighter ambient light
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2); // Brighter directional light
        directionalLight.position.set(5, 10, 7);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 1024;
        directionalLight.shadow.mapSize.height = 1024;
        scene.add(directionalLight);

        // Add a secondary light to brighten the dots
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.8);
        fillLight.position.set(-5, 5, -10);
        scene.add(fillLight);

        // Setup camera
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 10, 20);
        camera.lookAt(0, 0, 0);
        cameraRef.current = camera;

        // Initialize renderer with transparency
        const renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true, // Enable transparency
            preserveDrawingBuffer: true // Needed for some devices
        });
        renderer.setSize(width, height);
        renderer.setClearColor(0x000000, 0); // Transparent background
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.setPixelRatio(window.devicePixelRatio);

        // Critical: Position the canvas as a fixed overlay
        renderer.domElement.style.position = 'fixed';
        renderer.domElement.style.top = '0';
        renderer.domElement.style.left = '0';
        renderer.domElement.style.width = '100vw';
        renderer.domElement.style.height = '100vh';
        renderer.domElement.style.zIndex = '5'; // Layer above text but below UI
        renderer.domElement.style.pointerEvents = 'none'; // Allow clicks to pass through

        // Append to the body instead of the container for full overlay
        document.body.appendChild(renderer.domElement);

        // Add controls (optional)
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableZoom = false;
        controls.enablePan = false;
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enabled = false; // Disable orbit controls for cleaner integration

        // Initialize physics
        const world = new CANNON.World();
        world.gravity.set(0, -9.82 * 3, 0); // Increased gravity for faster settling
        world.broadphase = new CANNON.NaiveBroadphase();
        world.defaultContactMaterial.friction = 0.3;
        worldRef.current = world;

        // Create materials
        const groundMaterial = new CANNON.Material("groundMaterial");
        const diceMaterial = new CANNON.Material("diceMaterial");
        const wallMaterial = new CANNON.Material("wallMaterial");

        // Define contact properties
        const diceGroundContact = new CANNON.ContactMaterial(
            groundMaterial,
            diceMaterial,
            {
                friction: 0.4,
                restitution: 0.3,
            }
        );

        const diceWallContact = new CANNON.ContactMaterial(
            wallMaterial,
            diceMaterial,
            {
                friction: 0.0, // Less friction with walls
                restitution: 0.8, // More bounce with walls
            }
        );

        world.addContactMaterial(diceGroundContact);
        world.addContactMaterial(diceWallContact);

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

        // Calculate wall positions based on camera view
        // Project screen edges to world space at same distance as center
        const viewDistance = 20; // Match this with camera Z position
        const vFOV = camera.fov * Math.PI / 180;
        const visibleHeight = 2 * Math.tan(vFOV / 2) * viewDistance;
        const visibleWidth = visibleHeight * camera.aspect;

        const boundaryWidth = visibleWidth * 0.8; // Use 80% of visible width
        const boundaryHeight = visibleHeight * 0.8; // Use 80% of visible height
        const boundaryDepth = 10; // Depth of play area

        // Create invisible boundaries
        createBoundaryWall(world, boundaryWidth, 0.1, boundaryDepth, 0, boundaryHeight / 2, 0, wallMaterial); // top wall
        createBoundaryWall(world, boundaryWidth, 0.1, boundaryDepth, 0, -boundaryHeight / 2, 0, wallMaterial); // bottom wall
        createBoundaryWall(world, boundaryWidth, boundaryHeight, 0.1, 0, 0, -boundaryDepth / 2, wallMaterial); // back wall
        createBoundaryWall(world, boundaryWidth, boundaryHeight, 0.1, 0, 0, boundaryDepth / 2, wallMaterial); // front wall
        createBoundaryWall(world, 0.1, boundaryHeight, boundaryDepth, -boundaryWidth / 2, 0, 0, wallMaterial); // left wall
        createBoundaryWall(world, 0.1, boundaryHeight, boundaryDepth, boundaryWidth / 2, 0, 0, wallMaterial); // right wall

        // Handle window resize
        const handleResize = () => {
            const newWidth = window.innerWidth;
            const newHeight = window.innerHeight;

            camera.aspect = newWidth / newHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(newWidth, newHeight);

            // Update frozen dice positions when window resizes
            diceRef.current.forEach(dice => {
                if (dice.frozen) {
                    updateScreenPosition(dice);
                }
            });
        };

        window.addEventListener('resize', handleResize);

        // Handle scroll - update frozen dice positions
        const handleScroll = () => {
            diceRef.current.forEach(dice => {
                if (dice.frozen) {
                    updateScreenPosition(dice);
                }
            });
        };

        window.addEventListener('scroll', handleScroll);

        // Function to update a die's screen position
        const updateScreenPosition = (dice: typeof diceRef.current[0]) => {
            if (!dice.frozen || !dice.mesh || !camera) return;

            const vector = new THREE.Vector3(
                dice.screenPosition.x,
                dice.screenPosition.y,
                dice.screenPosition.z
            );

            vector.project(camera);

            // Convert to screen coordinates
            const x = (vector.x * 0.5 + 0.5) * window.innerWidth;
            const y = (-vector.y * 0.5 + 0.5) * window.innerHeight;

            // Apply transform to make the dice fixed to the screen
            dice.mesh.userData.screenX = x;
            dice.mesh.userData.screenY = y;
        };

        // Animation loop
        const animate = () => {
            animationRef.current = requestAnimationFrame(animate);

            // Update physics for non-frozen dice
            if (world) {
                world.step(1 / 60);

                // Update dice positions
                diceRef.current.forEach(dice => {
                    if (dice.frozen) {
                        // For frozen dice, maintain their screen position
                        if (dice.mesh.userData.screenX !== undefined) {
                            // Convert screen position back to world coordinates
                            const vector = new THREE.Vector3(
                                (dice.mesh.userData.screenX / window.innerWidth) * 2 - 1,
                                -(dice.mesh.userData.screenY / window.innerHeight) * 2 + 1,
                                0.5
                            );

                            vector.unproject(camera);
                            const dir = vector.sub(camera.position).normalize();
                            const distance = -camera.position.z / dir.z;
                            const pos = camera.position.clone().add(dir.multiplyScalar(distance));

                            dice.mesh.position.copy(pos);
                        }
                    } else {
                        // For non-frozen dice, update from physics
                        const position = dice.body.position;
                        const quaternion = dice.body.quaternion;

                        dice.mesh.position.set(position.x, position.y, position.z);
                        dice.mesh.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);

                        // Keep dice visible - constrain their position
                        if (position.x < -boundaryWidth / 2) {
                            dice.body.position.x = -boundaryWidth / 2 + 0.8;
                            dice.body.velocity.x = Math.abs(dice.body.velocity.x) * 0.8; // Bounce back
                        }
                        if (position.x > boundaryWidth / 2) {
                            dice.body.position.x = boundaryWidth / 2 - 0.8;
                            dice.body.velocity.x = -Math.abs(dice.body.velocity.x) * 0.8; // Bounce back
                        }
                        if (position.z < -boundaryDepth / 2) {
                            dice.body.position.z = -boundaryDepth / 2 + 0.8;
                            dice.body.velocity.z = Math.abs(dice.body.velocity.z) * 0.8; // Bounce back
                        }
                        if (position.z > boundaryDepth / 2) {
                            dice.body.position.z = boundaryDepth / 2 - 0.8;
                            dice.body.velocity.z = -Math.abs(dice.body.velocity.z) * 0.8; // Bounce back
                        }
                        if (position.y > boundaryHeight / 2) {
                            dice.body.position.y = boundaryHeight / 2 - 0.8;
                            dice.body.velocity.y = -Math.abs(dice.body.velocity.y) * 0.8; // Bounce back
                        }
                    }
                });

                // Check if dice have settled
                if (isRolling) {
                    const allSettled = diceRef.current.every(dice => {
                        if (dice.frozen) return true;

                        const velocity = dice.body.velocity;
                        const angularVelocity = dice.body.angularVelocity;

                        return (
                            Math.abs(velocity.x) < 0.1 &&
                            Math.abs(velocity.y) < 0.1 &&
                            Math.abs(velocity.z) < 0.1 &&
                            Math.abs(angularVelocity.x) < 0.05 &&
                            Math.abs(angularVelocity.y) < 0.05 &&
                            Math.abs(angularVelocity.z) < 0.05
                        );
                    });

                    if (allSettled) {
                        // Freeze dice in place on screen
                        diceRef.current.forEach(dice => {
                            if (!dice.frozen) {
                                dice.frozen = true;

                                // Store the current screen position
                                const vector = dice.mesh.position.clone();
                                vector.project(camera);

                                // Convert to screen coordinates and store
                                const x = (vector.x * 0.5 + 0.5) * window.innerWidth;
                                const y = (-vector.y * 0.5 + 0.5) * window.innerHeight;

                                dice.mesh.userData.screenX = x;
                                dice.mesh.userData.screenY = y;
                                dice.screenPosition = {
                                    x: dice.mesh.position.x,
                                    y: dice.mesh.position.y,
                                    z: dice.mesh.position.z
                                };

                                // Remove physics body as we don't need it anymore
                                if (worldRef.current) {
                                    worldRef.current.removeBody(dice.body);
                                }
                            }
                        });

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
            rollDice();
        }, 500);

        // Cleanup on unmount
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }

            window.removeEventListener('resize', handleResize);
            window.removeEventListener('scroll', handleScroll);

            if (renderer.domElement && renderer.domElement.parentNode) {
                renderer.domElement.parentNode.removeChild(renderer.domElement);
            }
        };
    }, []);

    // Create a boundary wall - physics only, no visual component
    const createBoundaryWall = (
        world: CANNON.World,
        width: number,
        height: number,
        depth: number,
        x: number,
        y: number,
        z: number,
        material: CANNON.Material
    ) => {
        // Physics body only - no visual mesh
        const halfExtents = new CANNON.Vec3(width / 2, height / 2, depth / 2);
        const wallShape = new CANNON.Box(halfExtents);
        const wallBody = new CANNON.Body({
            mass: 0,
            material: material,
            shape: wallShape
        });
        wallBody.position.set(x, y, z);
        world.addBody(wallBody);
    };

    // Create a die
    const createDice = (world: CANNON.World, scene: THREE.Scene, x: number, y: number, z: number) => {
        // Create die geometry
        const dieSize = 1.5; // Larger for better visibility
        const geometry = new THREE.BoxGeometry(dieSize, dieSize, dieSize);
        const material = new THREE.MeshStandardMaterial({
            color: 0x8b0000, // Dark red color for better contrast
            roughness: 0.3,
            metalness: 0.2
        });

        // Create die mesh
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        // Add die dots
        addDieDots(mesh, dieSize);

        // Create physics body
        const diceMaterial = new CANNON.Material("diceMaterial");
        const shape = new CANNON.Box(new CANNON.Vec3(dieSize / 2, dieSize / 2, dieSize / 2));
        const body = new CANNON.Body({
            mass: 1,
            material: diceMaterial,
            shape: shape,
            linearDamping: 0.1, // Add some air resistance
            angularDamping: 0.1 // Add some rotational resistance
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

        // Apply impulse - stronger force for more dramatic throw
        body.applyImpulse(
            new CANNON.Vec3(10 + Math.random() * 8, -1, 5 + Math.random() * 8),
            new CANNON.Vec3(body.position.x, body.position.y, body.position.z)
        );

        // Add to reference with frozen flag set to false
        diceRef.current.push({
            mesh,
            body,
            frozen: false,
            screenPosition: { x: 0, y: 0, z: 0 }
        });

        return { mesh, body };
    };

    // Add dots to a die face - as circles instead of spheres
    const addDieDots = (dieMesh: THREE.Mesh, dieSize: number) => {
        const dotSize = dieSize * 0.12; // Scale dots with die size
        const offset = dieSize * 0.5 + 0.01; // Position dots on the surface with slight offset
        const dotDistance = dieSize * 0.25; // Scaled spacing

        // Create circle geometry for dots
        const createCircleDot = (x: number, y: number, z: number, normal: THREE.Vector3) => {
            // Create a circle geometry
            const circleGeometry = new THREE.CircleGeometry(dotSize, 32);
            const circleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });

            // Create the mesh
            const circle = new THREE.Mesh(circleGeometry, circleMaterial);

            // Position it
            circle.position.set(x, y, z);

            // Orient it to face outward
            circle.lookAt(x + normal.x, y + normal.y, z + normal.z);

            // Add to die
            dieMesh.add(circle);
            return circle;
        };

        // Face 1 (single dot - front face)
        createCircleDot(0, 0, offset, new THREE.Vector3(0, 0, 1));

        // Face 2 (two dots - back face)
        createCircleDot(dotDistance, dotDistance, -offset, new THREE.Vector3(0, 0, -1));
        createCircleDot(-dotDistance, -dotDistance, -offset, new THREE.Vector3(0, 0, -1));

        // Face 3 (three dots - right face)
        createCircleDot(offset, dotDistance, dotDistance, new THREE.Vector3(1, 0, 0));
        createCircleDot(offset, 0, 0, new THREE.Vector3(1, 0, 0));
        createCircleDot(offset, -dotDistance, -dotDistance, new THREE.Vector3(1, 0, 0));

        // Face 4 (four dots - left face)
        createCircleDot(-offset, dotDistance, dotDistance, new THREE.Vector3(-1, 0, 0));
        createCircleDot(-offset, dotDistance, -dotDistance, new THREE.Vector3(-1, 0, 0));
        createCircleDot(-offset, -dotDistance, dotDistance, new THREE.Vector3(-1, 0, 0));
        createCircleDot(-offset, -dotDistance, -dotDistance, new THREE.Vector3(-1, 0, 0));

        // Face 5 (five dots - top face)
        createCircleDot(dotDistance, offset, dotDistance, new THREE.Vector3(0, 1, 0));
        createCircleDot(-dotDistance, offset, dotDistance, new THREE.Vector3(0, 1, 0));
        createCircleDot(0, offset, 0, new THREE.Vector3(0, 1, 0));
        createCircleDot(dotDistance, offset, -dotDistance, new THREE.Vector3(0, 1, 0));
        createCircleDot(-dotDistance, offset, -dotDistance, new THREE.Vector3(0, 1, 0));

        // Face 6 (six dots - bottom face)
        createCircleDot(dotDistance, -offset, dotDistance, new THREE.Vector3(0, -1, 0));
        createCircleDot(0, -offset, dotDistance, new THREE.Vector3(0, -1, 0));
        createCircleDot(-dotDistance, -offset, dotDistance, new THREE.Vector3(0, -1, 0));
        createCircleDot(dotDistance, -offset, -dotDistance, new THREE.Vector3(0, -1, 0));
        createCircleDot(0, -offset, -dotDistance, new THREE.Vector3(0, -1, 0));
        createCircleDot(-dotDistance, -offset, -dotDistance, new THREE.Vector3(0, -1, 0));
    };

    // Roll dice
    const rollDice = () => {
        if (isRolling || !worldRef.current || !sceneRef.current) return;

        setIsRolling(true);
        setShowButton(false);

        // Remove existing dice
        diceRef.current.forEach(die => {
            die.mesh.removeFromParent();
            if (!die.frozen && worldRef.current) {
                worldRef.current.removeBody(die.body);
            }
        });
        diceRef.current = [];

        // Create new dice - start from top left
        createDice(worldRef.current, sceneRef.current, -5, 10, -2);
        createDice(worldRef.current, sceneRef.current, -3, 12, 2);
    };

    return (
        <div className="inline-block relative" style={{ width: '100px', height: '100px' }}>
            {/* This is just a placeholder div that doesn't take up space */}
            <div
                ref={containerRef}
                className="pointer-events-none"
                style={{ width: '100%', height: '100%', position: 'absolute', zIndex: 1 }}
            />

            {showButton && !isRolling && (
                <button
                    onClick={rollDice}
                    className="absolute px-6 py-2 rounded-md font-bold text-white bg-blue-500 hover:bg-blue-600 transform transition-transform hover:scale-105"
                    style={{
                        bottom: '-40px',
                        left: '50%',
                        transform: 'translateX(-50%)',
                        zIndex: 100, // Very high z-index
                        pointerEvents: 'auto' // Make button clickable
                    }}
                >
                    Roll Dice
                </button>
            )}
        </div>
    );
};

export default DiceAnimation;