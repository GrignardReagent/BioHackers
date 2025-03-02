'use client';

import { Box, Button, Checkbox, FileInput, Image, Input, LoadingOverlay, SimpleGrid, Stack, Switch, Textarea } from "@mantine/core";
import { useEffect, useRef, useState } from "react";
import { BarChart } from '@mantine/charts';

export default function Home() {
  const [image, setImage] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');
  const [classCount, setClassCount] = useState<{ key: string; value: unknown }[]>([]);
  const [visible, setVisible] = useState(false);
  const [answer, setAnswer] = useState('');

  const question = 'Analyze the advantages and disadvantages of expanding a runway in this area for Heathrow Airport, and also suggest any potential mitigation strategies.';
  const answer_ = `Pros:
Improved Transportation and Connectivity: The new runway will enhance the transportation network, alleviating congestion and increasing international connectivity. It will support greater flight operations, boosting London's position as a global aviation hub.
Economic Growth: The early surge in employment opportunities will benefit construction workers, followed by long-term benefits through aviation-related industries. This will positively impact both residential (15%) and industrial (1%) areas, leading to job creation.
Development Potential: The expansion will boost local infrastructure and business activity, particularly in the residential (15%) and industrial (1%) zones, making the area more economically viable.
Cons:
Environmental Impact: Increased noise levels and air quality (AQI) will have a detrimental effect on both green belts (62%) and waterways (1%). The higher CO₂ emissions will impact biodiversity, with significant risks to the environment as the biodiversity index declines.
Traffic Congestion: The growth of traffic congestion, marked by a sharp increase, will impact residential areas, leading to urban strain, longer commute times, and lower quality of life.
Land Use: Conversion of farmland (21%) and green belts (62%) for expansion may lead to loss of agricultural land and natural habitats, contributing to environmental degradation.
Mitigation Strategies:
Noise and Air Pollution Control: Implement noise barriers and optimize flight paths to minimize disruption to residential and green belt areas. Use sustainable aircraft technologies to reduce emissions.
Ecological Restoration: Protect nearby waterways and green belts, and offset habitat loss with local conservation projects to maintain biodiversity.
Traffic Management: Improve local transport infrastructure, enhance public transport options, and consider congestion pricing to reduce traffic stress.
Community Support: Offer compensation or relocation options to residents in heavily affected areas, especially near noise-prone zones.`;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const file = formData.get('image') as File;

    if (file) {
      try {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImage(reader.result as string);
        };
        reader.readAsDataURL(file);

        setVisible(true);
        const response = await fetch('http://localhost:8000/statistic', {
          method: 'POST',
          body: formData,
        });

        setVisible(false);
        if (!response.ok) {
          throw new Error('File upload failed');
        } else {
          const data = await response.json();
          console.log(data);
          setClassCount(Object.entries(data).map(([key, value]) => ({ key, value })));
          console.log(classCount);
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
      }
    } else {
      alert('Please select a file to upload');
    }
  };

  return (
    <>
      <form className="m-3" onSubmit={handleSubmit}>
        <FileInput
          m={3}
          placeholder="Upload image"
          name="image"
        />
        <Button m={3} type="submit">Upload</Button>
      </form>
      <div className="m-3">
        <SimpleGrid cols={2}>
          <div style={{ overflow: 'hidden', borderRadius: '500px' }}>
            <Box style={{
              position: 'relative',
            }}>
              <Image
                src={image}
                alt="Background"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />
            </Box>
          </div>
          <div>
            <LoadingOverlay visible={visible} zIndex={1000} overlayProps={{ radius: "sm", blur: 2 }} />
            <BarChart
              data={classCount}
              color="blue"
              style={{ height: 300 }} dataKey={"key"} series={[{ name: 'value', color: 'violet.6' }]} />
          </div>
        </SimpleGrid>
      </div>
      <div className="m-3">
        <Input defaultValue={question}/>
        <Button onClick={() => setAnswer(answer_)} m={3}>Ask</Button>
        <Textarea placeholder="Answer" m={3} value={answer} readOnly rows={20} />
      </div>
    </>
  );
}
